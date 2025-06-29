import torch
import torch.nn as nn
import torch.nn.functional as F


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


class UNetResBlock_PosEnc(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode="circular"),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, padding_mode="circular"),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, padding_mode="circular"),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        # residual 用に in/out チャネル不一致を整える 1x1 Conv
        self.res_conv = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

        self.final_activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: [B, time_embed_dim]
        emb = self.time_mlp(t_emb).unsqueeze(-1).expand(-1, -1, x.size(-1))  # [B, in_channels, L]
        x_in = x + emb
        out = self.conv_block(x_in)
        res = self.res_conv(x)
        return self.final_activation(out + res)


class ConditionalResidualUNet_PosEnc(nn.Module):
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
        self.down1 = UNetResBlock_PosEnc(in_channels, 64, time_embed_dim)
        self.down2 = UNetResBlock_PosEnc(64, 128, time_embed_dim)
        self.down3 = UNetResBlock_PosEnc(128, 256, time_embed_dim)
        self.down4 = UNetResBlock_PosEnc(256, 512, time_embed_dim)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNetResBlock_PosEnc(512, 1024, time_embed_dim)

        # Projection convs after interpolation
        self.upproj4 = nn.Conv1d(1024, 512, kernel_size=1)
        self.upblock4 = UNetResBlock_PosEnc(512 * 2, 512, time_embed_dim)

        self.upproj3 = nn.Conv1d(512, 256, kernel_size=1)
        self.upblock3 = UNetResBlock_PosEnc(256 * 2, 256, time_embed_dim)

        self.upproj2 = nn.Conv1d(256, 128, kernel_size=1)
        self.upblock2 = UNetResBlock_PosEnc(128 * 2, 128, time_embed_dim)

        self.upproj1 = nn.Conv1d(128, 64, kernel_size=1)
        self.upblock1 = UNetResBlock_PosEnc(64 * 2, 64, time_embed_dim)

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
