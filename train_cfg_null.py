from dotenv import load_dotenv

load_dotenv()

import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------
# Weight & Biases のインポートと初期化
# ------------------------------------------
import wandb

wandb.login(key=os.environ["WANDB_API_KEY"])

# ハイパーパラメータなどの設定
execution_name = "252418-003"
num_epochs = 2000
initial_lr = 2e-4
b1 = 0.0
b2 = 0.9
batch_size = 32
diffusion_params = {"num_timesteps": 1000, "beta_start": 1e-4, "beta_end": 2e-2}
output_mode = "conv3x3"
guidance_scale = 3.0  # Classfier-Free Guidance スケール
p_uncond = 0.1

wandb.init(
    project="airfoil_diffusion",
    name=execution_name,
    config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "initial_lr": initial_lr,
        "b1": b1,
        "b2": b2,
        "diffusion": diffusion_params,
        "output_mode": output_mode,
        "guidance_scale": guidance_scale,
        "p_uncond": p_uncond,
        "dataset_prefix": "NACA&Joukowski",
        "memo": "optimizerをAdamに変更. Classifier-Free Guidance追加(null)",
    },
)
config = wandb.config


# ============================================================
# 1. モデルアーキテクチャの定義 (UNet) と LabelEmbedder, UNetConvBlock
# ============================================================
class LabelEmbedder(nn.Module):
    def __init__(self, c_dim: int = 1, t_dim: int = 1):
        super().__init__()
        # c 用 embed (無条件時に null_embed を置換)
        self.embed_c = nn.Sequential(
            nn.Linear(c_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 1)),
        )
        self.null_c = nn.Parameter(torch.randn(512, 1))
        # t 用 embed (常に有条件)
        self.embed_t = nn.Sequential(
            nn.Linear(t_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 1)),
        )

    def forward(self, c: torch.Tensor, t: torch.Tensor, uncond_mask: torch.Tensor = None) -> torch.Tensor:
        """
        c: (B,1), t: (B,1)
        uncond_mask: (B,) bool
        """
        # ← ここで float にキャスト
        c = c.float()
        t = t.float()

        B = c.size(0)
        emb_c = self.embed_c(c)  # (B,512,1)
        emb_t = self.embed_t(t)  # (B,512,1)
        if uncond_mask is not None:
            # null_c を展開
            null = self.null_c.unsqueeze(0).expand(B, -1, -1)
            mask = uncond_mask.view(B, 1, 1).float().to(c.device)
            emb_c = emb_c * (1 - mask) + null * mask
        # c と t の表現を足し合わせて返す
        return emb_c + emb_t


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int = 2, label_dim: int = 2, output_mode: str = "conv3x3"):
        """
        Args:
            in_channels (int): 入力データのチャネル数（例: 座標 x, y）
            label_dim (int): 条件情報（例: Cl と t）の次元数
        """
        super().__init__()
        self.output_mode = output_mode
        self.in_channels = in_channels

        # Embed label information (e.g. Cl, t)
        self.label_embedder = LabelEmbedder(c_dim=1, t_dim=1)

        # Downsampling blocks
        self.down1 = UNetConvBlock(in_channels, 64)
        self.down2 = UNetConvBlock(64, 128)
        self.down3 = UNetConvBlock(128, 256)
        self.down4 = UNetConvBlock(256, 512)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNetConvBlock(512, 1024)

        # Upsampling blocks
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.upblock4 = UNetConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.upblock3 = UNetConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.upblock2 = UNetConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.upblock1 = UNetConvBlock(128, 64)

        # Output Layer
        if output_mode == "conv1x1":
            self.output_layer = nn.Conv1d(64, in_channels, kernel_size=1)
        elif output_mode == "conv3x3":
            self.output_layer = nn.Conv1d(64, in_channels, kernel_size=3, padding=1, padding_mode="circular")
        elif output_mode == "fc":
            self.output_layer = nn.Linear(64 * 248, in_channels * 248)
        elif output_mode == "fc_nn":
            self.output_layer = nn.Sequential(
                nn.Linear(64 * 248, 2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, in_channels * 248),
            )
        else:
            raise ValueError(f"Invalid output_mode: {output_mode}")

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, uncond_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # t の次元調整＆dtype修正
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.float()
        c = c.float()

        # ラベル埋め込み
        label_embed = self.label_embedder(c, t, uncond_mask)  # (B,512,1)

        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(d4))

        # Upsample with label x skip
        u4 = self.upconv4(bottleneck)
        u4 = torch.cat([u4, label_embed], dim=2)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.upblock4(u4)

        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.upblock3(u3)

        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.upblock2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.upblock1(u1)

        if self.output_mode in ["conv1x1", "conv3x3"]:
            return self.output_layer(u1)
        else:
            B = x.size(0)
            out = u1.view(B, -1)
            out = self.output_layer(out)
            return out.view(B, self.in_channels, -1)


# ============================================================
# 2. Diffuserクラスの定義 (Diffusion Process, 逆拡散)
# ============================================================
class Diffuser:
    def __init__(self, num_timesteps=500, beta_start=1e-4, beta_end=2e-2, device="cpu", guidance_scale=1.0):
        self.num_timesteps = num_timesteps
        self.device = device
        self.guidance_scale = guidance_scale
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1, 1)
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, c):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alpha = self.alphas[t_idx].view(-1, 1, 1)
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1, 1)
        alpha_bar_prev = torch.ones_like(alpha_bar)
        mask = t > 1
        t_minus_2 = t[mask] - 2
        alpha_bar_prev[mask] = self.alpha_bars[t_minus_2].view(-1, 1, 1)

        # Classifier-Free Guidance:
        B = x.size(0)
        # 無条件時の mask
        uncond_mask = torch.ones(B, dtype=torch.bool, device=self.device)
        cond_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        model.eval()
        with torch.no_grad():
            eps_cond = model(x, c, t, uncond_mask=cond_mask)
            eps_uncond = model(x, c, t, uncond_mask=uncond_mask)
        model.train()

        eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0
        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + noise * std

    def generate_from_labels(self, model, labels, coord_shape=(2, 248)):
        batch_size = labels.size(0)
        x = torch.randn((batch_size, coord_shape[0], coord_shape[1]), device=self.device)
        for i in range(self.num_timesteps, 0, -1):
            t = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
            x = self.denoise(model, x, t, labels[:, 0:1])
        return x


# ============================================================
# 3. データローダー (AirfoilDataset) の読み込み
# ============================================================
class AirfoilDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coord_path="./dataset/NACA&Joukowski_coords_array.npy",
        cl_path="./dataset/NACA&Joukowski_cl_array.npy",
        norm_path="./dataset/NACA&Joukowski_normalization_stats.npz",
        normalize=True,
    ):
        coords_array = np.load(coord_path).astype(np.float32)  # shape: (N, 2, 248)
        cls_array = np.load(cl_path).astype(np.float32)[:, np.newaxis]  # shape: (N, 1)

        norm = np.load(norm_path)
        self.coord_mean = norm["coord_mean"]
        self.coord_std = norm["coord_std"]
        self.cl_mean = norm["cl_mean"][0]
        self.cl_std = norm["cl_std"][0]

        if normalize:
            coords_array = (coords_array - self.coord_mean) / self.coord_std
            cls_array = (cls_array - self.cl_mean) / self.cl_std

        self.coords_tensor = torch.tensor(coords_array, dtype=torch.float32)
        self.cls_tensor = torch.tensor(cls_array, dtype=torch.float32)
        self.normalize = normalize

    def __len__(self):
        return self.coords_tensor.shape[0]

    def __getitem__(self, idx):
        return self.coords_tensor[idx], self.cls_tensor[idx]

    def denormalize_coord(self, coord_tensor):
        std = torch.tensor(self.coord_std, dtype=torch.float32, device=coord_tensor.device)
        mean = torch.tensor(self.coord_mean, dtype=torch.float32, device=coord_tensor.device)
        return coord_tensor * std + mean

    def normalize_cl(self, cl_tensor):
        std = torch.tensor(self.cl_std, dtype=torch.float32, device=cl_tensor.device)
        mean = torch.tensor(self.cl_mean, dtype=torch.float32, device=cl_tensor.device)
        return (cl_tensor - mean) / std

    def denormalize_cl(self, cl_tensor):
        std = torch.tensor(self.cl_std, dtype=torch.float32, device=cl_tensor.device)
        mean = torch.tensor(self.cl_mean, dtype=torch.float32, device=cl_tensor.device)
        return cl_tensor * std + mean


# ============================================================
# 4. XFoilを用いたCL評価用関数 (get_cl)
# ============================================================

from xfoil import XFoil
from xfoil.model import Airfoil

# def get_cl(coord, xf=None, angle=5):
#     if xf is None:
#         xf = XFoil()
#         xf.print = False
#     xf.Re = 3e6
#     xf.max_iter = 100
#     datax, datay = coord.reshape(2, -1)
#     xf.airfoil = Airfoil(x=datax, y=datay)
#     c = xf.a(angle)
#     cl = c[0]
#     cl = np.round(cl, 10)
#     return cl

# グローバルにXFoilインスタンスを保持し、再利用する
_global_xf = None


def get_xf_instance():
    global _global_xf
    if _global_xf is None:
        _global_xf = XFoil()
        _global_xf.print = False
    return _global_xf


def get_cl(coord, xf=None, angle=5):
    if xf is None:
        xf = get_xf_instance()
    xf.Re = 3e6
    xf.max_iter = 100
    datax, datay = coord.reshape(2, -1)
    xf.airfoil = Airfoil(x=datax, y=datay)
    c = xf.a(angle)
    cl = c[0]
    cl = np.round(cl, 10)
    return cl


# ============================================================
# 5. 補助関数: running average filter, Convexity Loss, Smoothness Loss, CL Loss
# ============================================================
def running_average_filter(coord, kernel_size=9):
    x, y = coord
    x_filtered = np.convolve(x, np.ones(kernel_size) / kernel_size, mode="same")
    y_filtered = np.convolve(y, np.ones(kernel_size) / kernel_size, mode="same")
    return np.array([x_filtered, y_filtered])


def convexity_loss(coord):
    x, y = coord
    dx = np.diff(x)
    dy = np.diff(y)
    vectors = np.stack([dx, dy], axis=1)
    angles = []
    for k in range(len(vectors) - 1):
        v1 = vectors[k]
        v2 = vectors[k + 1]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            angle = 0
        else:
            dot = np.dot(v1, v2)
            cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
        angles.append(angle)
    return np.sum(angles) / len(angles) if len(angles) > 0 else 0


def smoothness_loss(coord):
    filtered = running_average_filter(coord, kernel_size=9)
    return np.mean((coord - filtered) ** 2)


def cl_loss_function(cl_conditioned, cl_evaluated):
    return np.mean((cl_conditioned - cl_evaluated) ** 2)


def evaluate_generated_samples(samples, conditioned_cls, dataset):
    convexity_losses = []
    smoothness_losses = []
    cl_losses = []
    convergence_count_raw = 0
    convergence_count_strict = 0

    total_count = 0
    for i in range(samples.shape[0]):
        sample = samples[i].detach().cpu()
        sample_denorm = dataset.denormalize_coord(sample).cpu().numpy()
        try:
            cl_eval = get_cl(sample_denorm, angle=5)
        except Exception as _:
            cl_eval = None
        convexity_l = convexity_loss(sample_denorm)
        smoothness_l = smoothness_loss(sample_denorm)
        if cl_eval is not None and not np.isnan(cl_eval):
            cl_loss_val = cl_loss_function(conditioned_cls[i], cl_eval)
            converged = True
            convergence_count_raw += 1
        else:
            cl_loss_val = np.nan
            converged = False
        convexity_losses.append(convexity_l)
        smoothness_losses.append(smoothness_l)
        cl_losses.append(cl_loss_val)
        if (convexity_l < 0.1 and smoothness_l < 0.1 and cl_loss_val < 0.1) and converged:
            convergence_count_strict += 1
        total_count += 1
    avg_convexity_loss = np.mean(convexity_losses)
    avg_smoothness_loss = np.mean(smoothness_losses)
    if np.isnan(cl_losses).all() or len(cl_losses) == 0:
        avg_cl_loss = 0
    else:
        avg_cl_loss = np.nanmean(cl_losses)
    cl_convergence_ratio_raw = convergence_count_raw / total_count if total_count > 0 else np.nan
    cl_convergence_ratio_strict = convergence_count_strict / total_count if total_count > 0 else np.nan
    return avg_convexity_loss, avg_smoothness_loss, avg_cl_loss, cl_convergence_ratio_raw, cl_convergence_ratio_strict


# ============================================================
# 6. 結果保存用のディレクトリ作成
# ============================================================
output_dirs = {
    "model_info": f"./results/{execution_name}/model_info",
    "training_metrics": f"./results/{execution_name}/training_metrics",
    "evaluation_metrics": f"./results/{execution_name}/evaluation_metrics",
    "samples": f"./results/{execution_name}/samples",
    "weights": f"./results/{execution_name}/weights",
}
for folder in output_dirs.values():
    os.makedirs(folder, exist_ok=True)

# ============================================================
# 7. 学習ループ・初期設定
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
wandb.config.update({"device": str(device)})
dataset = AirfoilDataset(normalize=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = ConditionalUNet(in_channels=2, label_dim=2, output_mode=output_mode).to(device)
diffuser = Diffuser(
    num_timesteps=diffusion_params["num_timesteps"],
    beta_start=diffusion_params["beta_start"],
    beta_end=diffusion_params["beta_end"],
    device=device,
    guidance_scale=guidance_scale,
)

# optimizer = optim.SGD(model.parameters(), lr=initial_lr)
optimizer = optim.Adam(model.parameters(), lr=initial_lr, betas=(b1, b2))

# 最初に1回だけモデルパラメータ数とモデルサイズ(MB)を保存
model_param_count = sum(p.numel() for p in model.parameters())
model_param_count_path = os.path.join(output_dirs["model_info"], "model_parameter_count.txt")
with open(model_param_count_path, "w") as f:
    f.write(f"Model Parameter Count: {model_param_count}\n")
model_size_MB = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6
with open(os.path.join(output_dirs["model_info"], "model_size_MB.txt"), "w") as f:
    f.write(str(model_size_MB))
print(f"初回保存: モデルパラメータ数: {model_param_count}, モデルサイズ: {model_size_MB:.2f} MB")

# モデルサイズなどの静的情報も wandb に記録
wandb.config.update({"model_param_count": model_param_count})
wandb.config.update({"model_size_MB": model_size_MB})

# ============================================================
# 8. 学習ループ
# ============================================================
evaluation_interval = 200  # default:200

train_loss_history = []

# 各評価指標ごとの時系列履歴 (今後の評価ブロックで利用)
eval_history = {
    "convexity_loss_mean": [],
    "smoothness_loss_mean": [],
    "cl_loss_mean": [],
    "cl_convergence_ratio_raw": [],
    "cl_convergence_ratio_strict": [],
    # "MWT_sampling_sec": [],
    f"EP_time_sec_for_last_{evaluation_interval}epochs": [],
}

prev_eval_time = time.time()

print("学習開始...")

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_losses = []
    for x, cl in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
        # for x, cl in loader:
        x = x.to(device)
        cl = cl.to(device)
        t = torch.randint(low=1, high=diffuser.num_timesteps + 1, size=(x.size(0),), device=device)
        x_t, noise = diffuser.add_noise(x, t)

        # 無条件化マスク
        mask = torch.rand(x.size(0), device=device) < p_uncond
        # c,t は正規化済み
        noise_pred = model(x_t, cl, t.unsqueeze(1), uncond_mask=mask)

        loss = nn.MSELoss()(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_epoch_loss = np.mean(epoch_losses)
    train_loss_history.append(avg_epoch_loss)
    # print(f"Epoch {epoch}/{num_epochs} Loss: {avg_epoch_loss:.6f}")

    # 学習損失などを wandb に記録
    wandb.log({"train_loss": avg_epoch_loss, "epoch": epoch})

    lr_new = initial_lr * ((1 - epoch / num_epochs) ** 0.4)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_new

    # ローカルにも学習履歴を保存
    with open(os.path.join(output_dirs["training_metrics"], "training_loss_history.txt"), "a") as f:
        f.write(f"{epoch},{avg_epoch_loss}\n")

    # if epoch % evaluation_interval == 0 or epoch == num_epochs:
    #     plt.figure()
    #     plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, marker="o")
    #     plt.yscale("log")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Diffusion Loss (log scale)")
    #     plt.title("Training Diffusion Loss")
    #     plt.tight_layout()
    #     loss_plot_path = os.path.join(output_dirs["training_metrics"], f"loss_epoch_{epoch}.png")
    #     plt.savefig(loss_plot_path)
    #     plt.close()
    #     # wandb に画像をアップロード
    #     wandb.log({"loss_plot": wandb.Image(loss_plot_path), "epoch": epoch})

    if epoch % evaluation_interval == 0:
        with torch.no_grad():
            eval_start = time.time()
            print(f"--- Evaluation at epoch {epoch} ---")
            cl_eval_values = np.linspace(0.5, 1.2, 71)
            num_samples_for_each_cl = 10  # default:10
            convexity_list = []
            smoothness_list = []
            cl_loss_list = []
            convergence_ratios_raw = []
            convergence_ratios_strict = []
            for cl_val in tqdm(cl_eval_values, total=len(cl_eval_values)):
                # for cl_val in cl_eval_values:
                cond = torch.tensor([[cl_val, 0.0]] * num_samples_for_each_cl, dtype=torch.float32, device=device)
                cond_norm = dataset.normalize_cl(cond)
                generated = diffuser.generate_from_labels(model, cond_norm, coord_shape=(2, 248))
                cl_conditioned = np.array([cl_val] * num_samples_for_each_cl)
                conv_l, smooth_l, cl_l, conv_ratio_raw, conv_ratio_strict = evaluate_generated_samples(
                    generated, cl_conditioned, dataset
                )
                convexity_list.append(conv_l)
                smoothness_list.append(smooth_l)
                cl_loss_list.append(cl_l)
                convergence_ratios_raw.append(conv_ratio_raw)
                convergence_ratios_strict.append(conv_ratio_strict)

            eval_metrics = {
                "convexity_loss_mean": np.mean(convexity_list),
                "smoothness_loss_mean": np.mean(smoothness_list),
                "cl_loss_mean": np.mean(cl_loss_list),
                "cl_convergence_ratio_raw": np.mean(convergence_ratios_raw),
                "cl_convergence_ratio_strict": np.mean(convergence_ratios_strict),
            }

            # # MWT Sampling: 1サンプル生成の平均wall time (例として100サンプル)
            # sample_times = []
            # num_sample = 100  # default:100
            # cond = torch.tensor([[0.8, 0.0]] * 1, dtype=torch.float32, device=device)
            # cond_norm = dataset.normalize_cl(cond)
            # for _ in range(num_sample):
            #     start_t = time.time()
            #     _ = diffuser.generate_from_labels(model, cond_norm, coord_shape=(2, 248))
            #     sample_times.append(time.time() - start_t)
            # eval_metrics["MWT_sampling_sec"] = np.mean(sample_times)

            # ep_time = time.time() - prev_eval_time
            # prev_eval_time = time.time()
            # eval_metrics[f"EP_time_sec_for_last_{evaluation_interval}epochs"] = ep_time

            # 各指標を wandb に記録
            wandb.log({**eval_metrics, "epoch": epoch})

            # # 各指標ごとの履歴保存と画像作成
            # for key, value in eval_metrics.items():
            #     eval_history[key].append((epoch, value))
            #     epochs_list, values_list = zip(*eval_history[key])
            #     plt.figure()
            #     plt.plot(epochs_list, values_list, marker="o")
            #     plt.xlabel("Epoch")
            #     plt.ylabel(key)
            #     plt.title(f"{key} over Evaluations (up to epoch {epoch})")
            #     plt.tight_layout()
            #     metric_plot_path = os.path.join(output_dirs["evaluation_metrics"], f"{key}_epoch_{epoch}.png")
            #     plt.savefig(metric_plot_path)
            #     plt.close()
            #     print(f"{key}の評価グラフ保存: {metric_plot_path}")
            #     # wandb に画像をアップロード
            #     wandb.log({f"{key}_plot": wandb.Image(metric_plot_path), "epoch": epoch})
            #     data_save_path = os.path.join(output_dirs["evaluation_metrics"], f"{key}_data_epoch_{epoch}.txt")
            #     with open(data_save_path, "w") as f:
            #         f.write("Epoch,Value\n")
            #         for e, val in eval_history[key]:
            #             f.write(f"{e},{val}\n")
            #     print(f"{key}の生データ保存: {data_save_path}")

            # 8条件 (CL = [0.5,0.6,...,1.2]) でのサンプルプロット (5サンプルずつ, 5行×8列)
            cl_plot_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
            fig, axs = plt.subplots(5, len(cl_plot_values), figsize=(20, 12))
            for col, cl_val in tqdm(enumerate(cl_plot_values), total=len(cl_plot_values)):
                cond = torch.tensor([[cl_val, 0.0]] * 5, dtype=torch.float32, device=device)  # default:5
                cond_norm = dataset.normalize_cl(cond)
                generated = diffuser.generate_from_labels(model, cond_norm, coord_shape=(2, 248))
                for row in range(5):  # default:5
                    sample = generated[row].detach().cpu()
                    sample_denorm = dataset.denormalize_coord(sample).cpu().numpy()
                    x_coord = sample_denorm[0, :]
                    y_coord = sample_denorm[1, :]
                    try:
                        cl_eval = get_cl(sample_denorm, angle=5)
                    except Exception as _:
                        cl_eval = np.nan
                    conv_loss = convexity_loss(sample_denorm)
                    axs[row, col].plot(x_coord, y_coord)
                    axs[row, col].set_title(f"CL: {cl_eval:.2f}\nConv: {conv_loss:.3f}", fontsize=8)
                    axs[row, col].tick_params(labelsize=6)
                    axs[row, col].grid(True)
            plt.tight_layout()
            sample_plot_path = os.path.join(output_dirs["samples"], f"samples_epoch_{epoch}.png")
            plt.savefig(sample_plot_path)
            plt.close()
            print(f"生成サンプルプロット保存: {sample_plot_path}")
            wandb.log({"generated_samples": wandb.Image(sample_plot_path), "epoch": epoch})

            # ★ {evaluation_interval}epoch毎に中間モデルの重みを保存 (拡張子 .pth)
            intermediate_model_path = os.path.join(output_dirs["weights"], f"model_weights_epoch_{epoch}.pth")
            torch.save(model.state_dict(), intermediate_model_path)
            print(f"Epoch {epoch}: 中間モデルの重み保存: {intermediate_model_path}")
            wandb.save(intermediate_model_path)

        # 評価ブロック終了後、ガベージコレクションと GPU キャッシュの解放
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 9. 学習終了後，最終モデルの重みを保存
# ============================================================
final_model_path = os.path.join(output_dirs["weights"], "final_model_weights.pt")
torch.save(model.state_dict(), final_model_path)
print(f"学習終了．最終モデルの重み保存: {final_model_path}")
wandb.save(final_model_path)
wandb.finish()
