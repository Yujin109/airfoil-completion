import datetime
import math

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ハイパーパラメータなどの設定
execution_name = "250426-001:CFG(0)+pos-enc+beta_end0.05"
num_epochs = 2000
initial_lr = 2e-4
b1 = 0.0
b2 = 0.9
batch_size = 32
diffusion_params = {
    "num_timesteps": 500,
    "beta_start": 1e-4,
    "beta_end": 5e-2,
    "beta_schedule": "linear",  # "linear" or "cosine"
    "cosine_s": 0,  # default=0.008
}
output_mode = "conv3x3"
guidance_scale = 2.0  # Classfier-Free Guidance スケール
p_uncond = 0.1
evaluation_interval = 500  # default:200


# ============================================================
# 1. モデルアーキテクチャの定義 (UNet)
# ============================================================


def pos_encoding(timesteps: torch.Tensor, output_dim: int) -> torch.Tensor:
    """
    ベクトル化された positional encoding (sin/cos) を GPU 上で高速に計算
    timesteps: Tensor of shape [B, 1] or [B]  dtype torch.float32 or torch.int64
    output_dim: embedding dimension
    returns: Tensor [B, output_dim]
    """
    if timesteps.dim() == 1:
        timesteps = timesteps.unsqueeze(1).to(torch.float32)
    else:
        timesteps = timesteps.to(torch.float32)
    device = timesteps.device
    # B = timesteps.size(0)
    # 角度レートの計算
    dims = torch.arange(output_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (dims / output_dim))  # shape [output_dim]
    # timesteps を broadcast
    angles = timesteps * inv_freq.unsqueeze(0)  # shape [B, output_dim]
    # sin を偶数 idx, cos を奇数 idx に適用
    pe = torch.empty_like(angles)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class UNetConvBlock_PosEnc(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(inplace=True),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: [B, time_embed_dim]
        emb = self.time_mlp(t_emb).unsqueeze(-1).expand(-1, -1, x.size(-1))  # [B, in_channels, L]
        return self.block(x + emb)


class ConditionalUNet_PosEnc(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        label_dim: int = 1,
        time_embed_dim: int = 100,
        output_mode: str = "conv3x3",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.output_mode = output_mode

        # ラベル埋め込み (Cl と t)
        self.label_embedder = nn.Sequential(
            nn.Linear(label_dim, time_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Downsampling blocks
        self.down1 = UNetConvBlock_PosEnc(in_channels, 64, time_embed_dim)
        self.down2 = UNetConvBlock_PosEnc(64, 128, time_embed_dim)
        self.down3 = UNetConvBlock_PosEnc(128, 256, time_embed_dim)
        self.down4 = UNetConvBlock_PosEnc(256, 512, time_embed_dim)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNetConvBlock_PosEnc(512, 1024, time_embed_dim)

        # Projection convs after interpolation
        self.upproj4 = nn.Conv1d(1024, 512, kernel_size=1)
        self.upblock4 = UNetConvBlock_PosEnc(512 * 2, 512, time_embed_dim)

        self.upproj3 = nn.Conv1d(512, 256, kernel_size=1)
        self.upblock3 = UNetConvBlock_PosEnc(256 * 2, 256, time_embed_dim)

        self.upproj2 = nn.Conv1d(256, 128, kernel_size=1)
        self.upblock2 = UNetConvBlock_PosEnc(128 * 2, 128, time_embed_dim)

        self.upproj1 = nn.Conv1d(128, 64, kernel_size=1)
        self.upblock1 = UNetConvBlock_PosEnc(64 * 2, 64, time_embed_dim)

        # Output layer
        if output_mode == "conv1x1":
            self.output_layer = nn.Conv1d(64, in_channels, kernel_size=1)
        elif output_mode == "conv3x3":
            self.output_layer = nn.Conv1d(64, in_channels, kernel_size=3, padding=1, padding_mode="circular")
        elif output_mode in ["fc", "fc_nn"]:
            flat_dim = 64 * 248
            if output_mode == "fc":
                self.output_layer = nn.Linear(flat_dim, in_channels * 248)
            else:
                self.output_layer = nn.Sequential(
                    nn.Linear(flat_dim, 2048),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(2048, 2048),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(2048, 2048),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(2048, 1024),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(1024, in_channels * 248),
                )
        else:
            raise ValueError(f"Invalid output_mode: {output_mode}")

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        uncond_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # t, c を float tensor に
        t = t.view(-1, 1).to(x.dtype)
        c = c.to(x.dtype)

        # Positional encoding + label embedding
        t_emb = pos_encoding(t, self.time_embed_dim)  # [B, time_embed_dim]
        t_emb = t_emb.to(x.device)
        c_emb = self.label_embedder(c.to(x.device))
        if uncond_mask is not None:
            # uncond_mask==True のサンプルについて c_emb を 0 に
            mask = (~uncond_mask).to(x.dtype).unsqueeze(1)  # [B, 1]
            c_emb = c_emb * mask  # broadcasting で [B, time_embed_dim]
        # 条件付き or 無条件の埋め込みを合成
        t_emb = t_emb + c_emb

        # Downsampling
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)
        d4 = self.down4(self.pool(d3), t_emb)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(d4), t_emb)

        # Upsampling + skip connections via interpolation
        u4 = F.interpolate(bottleneck, size=d4.size(-1), mode="linear", align_corners=False)
        u4 = self.upproj4(u4)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.upblock4(u4, t_emb)

        u3 = F.interpolate(u4, size=d3.size(-1), mode="linear", align_corners=False)
        u3 = self.upproj3(u3)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.upblock3(u3, t_emb)

        u2 = F.interpolate(u3, size=d2.size(-1), mode="linear", align_corners=False)
        u2 = self.upproj2(u2)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.upblock2(u2, t_emb)

        u1 = F.interpolate(u2, size=d1.size(-1), mode="linear", align_corners=False)
        u1 = self.upproj1(u1)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.upblock1(u1, t_emb)

        # Output
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
    def __init__(
        self,
        num_timesteps: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu",
        guidance_scale: float = 1.0,
        beta_schedule: str = "linear",  # "linear" or "cosine"
        cosine_s: float = 0.008,  # cosine スケジュールのシフト量
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.guidance_scale = guidance_scale

        # β の生成
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps, s=cosine_s).to(device)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        # α, ᾱ を計算
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Improved DDPM の論文で提案された Cosine スケジュールです。
        ᾱ_t = cos(((t/T + s) / (1+s)) * π/2)^2 を用い、
        β_t = 1 - ᾱ_t / ᾱ_{t-1} で定義します。
        """
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)
        # ᾱ の前段階
        alphas_cumprod = torch.cos(((t / num_timesteps + s) / (1 + s)) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # β を差分で求める
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        # 数値的に不安定な極端な値をクリップ
        return torch.clamp(betas, max=0.999)

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


def compute_generation_diversity(generated_batch):
    """
    Args:
        generated_batch: torch.Tensor of shape [N, 2, 248]
    Returns:
        diversity: float, 分散値 (L2ノルムベース)
    """
    N = generated_batch.size(0)
    flattened = generated_batch.view(N, -1)  # [N, 496]
    mean_vector = torch.mean(flattened, dim=0, keepdim=True)  # [1, 496]
    diffs = flattened - mean_vector  # [N, 496]
    diversity = torch.mean(torch.norm(diffs, dim=1) ** 2).item()
    return diversity


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


# # ============================================================
# # 6. 結果保存用のディレクトリ作成
# # ============================================================
# output_dirs = {
#     "model_info": f"./results/{execution_name}/model_info",
#     "training_metrics": f"./results/{execution_name}/training_metrics",
#     "evaluation_metrics": f"./results/{execution_name}/evaluation_metrics",
#     "samples": f"./results/{execution_name}/samples",
#     "weights": f"./results/{execution_name}/weights",
# }
# for folder in output_dirs.values():
#     os.makedirs(folder, exist_ok=True)


# --- Diffuser に RePaint メソッドを追加 ---
class RePaintDiffuser(Diffuser):

    def __init__(self, *args, num_resampling=3, jump_length=10, **kwargs):
        super().__init__(*args, **kwargs)

        assert jump_length >= 1 and num_resampling >= 1
        self.num_resampling = num_resampling  # R: 同じtでのサンプリング回数（R=1だとJによらず通常の生成）
        self.jump_length = jump_length  # J: tを進める長さ（通常の生成ではJ=1）

    def add_noise_onestep(self, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all(), "t must be in [1, T]"
        t_idx = t - 1
        beta = self.betas[t_idx].view(-1, 1, 1)
        noise = torch.randn_like(x, device=self.device)
        x_t = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
        return x_t

    def schedule_timesteps(self, num_timesteps: int, jump_length: int, num_resampling: int):

        # RePaint論文内の実装 ###
        jumps = {}
        for j in range(0, num_timesteps - jump_length, jump_length):
            jumps[j] = num_resampling - 1
        t = num_timesteps
        ts = []

        while t >= 1:
            t = t - 1
            ts.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    ts.append(t)
        ts.append(-1)
        #######################

        # 全てのtを1ずつ増加させる
        ts = [t + 1 for t in ts]

        return ts

    def repaint(self, model, x_orig, mask, c):
        """
        RePaint アルゴリズムによる欠損値補完
        x_orig: [B,2,L] の既知領域データ
        mask:    [B,L] の bool Tensor (True=known, False=to inpaint)
        c:       [B,1] の正規化済み Cl 条件
        返り値:  [B,2,L] の inpainted サンプル
        """

        B, C, L = x_orig.shape
        device = x_orig.device
        # 初期ノイズ
        x = torch.randn_like(x_orig, device=device)
        mask = mask.unsqueeze(1).expand(-1, C, -1)

        ts = self.schedule_timesteps(self.num_timesteps, self.jump_length, self.num_resampling)

        for i, t in tqdm(enumerate(ts), desc="RePaint", total=len(ts)):
            # 現在のtをテンソル化
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            # 次のtを見て、tを進めるかor戻すかを決定
            if i < len(ts) - 1:
                next_t = ts[i + 1]
                if next_t == 0:
                    x_unknown = self.denoise(model=model, x=x, t=t_tensor, c=c)
                    x = torch.where(mask, x_orig, x_unknown)
                elif next_t < t:
                    # backward jump
                    x_known, _ = self.add_noise(x_0=x_orig, t=t_tensor - 1)
                    x_unknown = self.denoise(model=model, x=x, t=t_tensor, c=c)
                    x = torch.where(mask, x_known, x_unknown)
                elif next_t > t:
                    # forward jump
                    x = self.add_noise_onestep(x, t_tensor)
                else:
                    pass
            # else:
            #     # 最後のtはbackward jump
            #     x_known, _ = self.add_noise(x_0=x_orig, t=t - 1)
            #     x_unknown = self.denoise(model=model, x=x, t=t_tensor, c=c)
            #     x = torch.where(mask, x_known, x_unknown)

        return x


# mask function ---------------
#
def mask_fn_manual():
    # maskはshape=(248,)で、True=既知(known)、False=欠損(unknown)

    # 末尾を欠損しているパターン
    # mask = np.array([False] * 10 + [True] * 228 + [False] * 10)
    # 先頭を欠損しているパターン
    mask = np.array([True] * 114 + [False] * 20 + [True] * 114)
    # 中央上部を欠損しているパターン
    # mask = np.array([True] * 124 + [True] * 52 + [False] * 20 + [True] * 52)

    assert mask.shape == (248,)

    return mask


# config parameters ---------------
clusters: list = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.2)]
samples_per_cluster: int = 5
pattern: str = "known_vs_inpaint"  # 'known_vs_inpaint' or 'original_and_inpaint'
figsize: tuple = (20, 12)

num_resampling = 1
jump_length = 1
mask_fn = mask_fn_manual

seed = 42
# シード固定
if seed is not None:
    np.random.seed(seed)

# -----------------------------------


dataset = AirfoilDataset(normalize=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ConditionalUNet(in_channels=2, label_dim=2, output_mode=output_mode).to(device)
# model = ConditionalResidualUNet(in_channels=2, label_dim=2, output_mode=output_mode).to(device)
model = ConditionalUNet_PosEnc(
    in_channels=2,
    label_dim=1,
    time_embed_dim=100,
    output_mode=output_mode,
).to(device)
# model = ConditionalResidualUNet_PosEnc(
#     in_channels=2,
#     label_dim=1,
#     time_embed_dim=100,
#     output_mode=output_mode,
# ).to(device)

weight_path = "./results/250426-001:CFG(0)+pos-enc+beta_end0.05/weights/final_model_weights.pt"
checkpoint = torch.load(
    weight_path,
    map_location=torch.device("cpu"),  # ← ここで CPU にマッピング
)
model.load_state_dict(checkpoint)

diffuser = RePaintDiffuser(
    num_timesteps=diffusion_params["num_timesteps"],
    beta_start=diffusion_params["beta_start"],
    beta_end=diffusion_params["beta_end"],
    device=device,
    guidance_scale=guidance_scale,
    beta_schedule=diffusion_params["beta_schedule"],
    cosine_s=diffusion_params["cosine_s"],
    num_resampling=num_resampling,
    jump_length=jump_length,
)

# 保存先ディレクトリ作成
save_dir = "./repaint/images"
os.makedirs(save_dir, exist_ok=True)

# モデル名・重み名・日付・設定取得
model_name = type(model).__name__
weight_name = os.path.splitext(os.path.basename(weight_path))[0]
date_str = datetime.datetime.now().strftime("%y%m%d")  # 250430形式
R = diffuser.num_resampling
J = diffuser.jump_length

# クラスタごとのインデックスを取得
all_cl = dataset.cls_tensor.numpy().flatten()
original_cl = dataset.denormalize_cl(dataset.cls_tensor).numpy().flatten()

# 各クラスタに対応するサンプルID
selected_indices = []
for lo, hi in clusters:
    idxs = np.where((original_cl >= lo) & (original_cl < hi))[0]
    if len(idxs) < samples_per_cluster:
        raise ValueError(f"クラスタ [{lo},{hi}) に足りないサンプル: {len(idxs)} < {samples_per_cluster}")
    selected = np.random.choice(idxs, samples_per_cluster, replace=False)
    selected_indices.extend(selected.tolist())

B = len(selected_indices)

# データ準備
coords_batch = dataset.coords_tensor[selected_indices].to(diffuser.device)
# cl_batch = dataset.cls_tensor[selected_indices].to(diffusser.device)
cl_batch = (dataset.cls_tensor[selected_indices] + 0.1 / dataset.cl_std).to(diffuser.device)

# RePaint 実行
masks = torch.stack([torch.from_numpy(mask_fn()) for _ in coords_batch], dim=0).to(diffuser.device)
repainted = diffuser.repaint(model, coords_batch, masks, cl_batch)

# Denormalize for evaluation and plotting
orig_denorm = dataset.denormalize_coord(coords_batch).cpu().numpy()
rep_denorm = dataset.denormalize_coord(repainted).cpu().detach().numpy()
cl_rep_list = []

for coord in rep_denorm:
    cl_rep_list.append(get_cl(coord, angle=5))

# プロット
fig, axes = plt.subplots(samples_per_cluster, len(clusters), figsize=figsize)
for i, idx in enumerate(selected_indices):
    ci = i // samples_per_cluster
    si = i % samples_per_cluster
    ax = axes[si, ci]
    orig = orig_denorm[i]
    rep = rep_denorm[i]
    # masks[i] は shape=(L,) の bool 配列
    m = masks[i].cpu().numpy()
    # プロット
    # if pattern == "known_vs_inpaint":
    #     ax.plot(orig[0, m], orig[1, m], linestyle="-", label="known")
    #     ax.plot(rep[0, ~m], rep[1, ~m], linestyle="--", label="inpaint")
    # else:
    #     ax.plot(orig[0], orig[1], linestyle="-", label="original", color="")
    #     ax.plot(rep[0, ~m], rep[1, ~m], linestyle="--", label="inpaint")
    ax.plot(
        np.ma.array(orig[0], mask=~m), np.ma.array(orig[1], mask=~m), label="known", linestyle="-", color="tab:blue"
    )
    ax.plot(
        np.ma.array(orig[0], mask=m),
        np.ma.array(orig[1], mask=m),
        label="known_missing",
        linestyle="--",
        color="tab:blue",
    )
    ax.plot(
        np.ma.array(rep[0], mask=m), np.ma.array(rep[1], mask=m), label="inpaint", linestyle="-", color="tab:orange"
    )

    # タイトル
    cl_input = original_cl[idx]
    cl_out = cl_rep_list[i]
    ax.set_title(f"Cl_in={cl_input:.3f}\nCl_out={cl_out:.3f}")
    ax.tick_params(labelsize=6)
    ax.grid(True)
    # ax.axis("off")
plt.tight_layout()

# 画像保存
fname = f"{model_name}_{weight_name}_{date_str}_R{R}_J{J}.png"
fpath = os.path.join(save_dir, fname)
fig.savefig(fpath, dpi=500)
print(f"[Saved] {fpath}")

# plt.show()

print("OK!")
