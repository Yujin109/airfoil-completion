import math

import torch
from tqdm import tqdm


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
        model_name=None,
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.guidance_scale = guidance_scale
        self.model_name = model_name

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

        if self.model_name == "BaselineConditionalUNet":
            model.eval()
            with torch.no_grad():
                eps = model(x, c, t)
            model.train()
        elif self.model_name == "BaselineConditionalUNet_CFG_zero":
            # Classifier-Free Guidance
            c_uncond = torch.zeros_like(c)
            model.eval()
            with torch.no_grad():
                eps_cond = model(x, c, t)
                eps_uncond = model(x, c_uncond, t)
            model.train()
            eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        else:
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
        return x
