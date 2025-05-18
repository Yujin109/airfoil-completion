import torch
import torch.nn as nn


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


class UNetResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # ショートカット用の１×１ Conv（チャネル合わせ）
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )
        # 残差経路
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
        )
        # 最終活性化関数
        self.final_act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_act(self.block(x) + self.skip(x))


class ConditionalResidualUNet(nn.Module):
    def __init__(self, in_channels: int = 2, label_dim: int = 2, output_mode: str = "conv3x3"):
        """
        Args:
            in_channels (int): 入力データのチャネル数（例: 座標 x, y）
            label_dim (int): 条件情報（例: Cl と t）の次元数
            output_mode (str): 出力モード ("conv1x1", "conv3x3", "fc", "fc_nn")
        """
        super().__init__()
        self.output_mode = output_mode
        self.in_channels = in_channels

        # Embed label information (e.g. Cl, t)
        self.label_embedder = LabelEmbedder(c_dim=1, t_dim=1)

        # Downsampling blocks
        self.down1 = UNetResBlock(in_channels, 64)
        self.down2 = UNetResBlock(64, 128)
        self.down3 = UNetResBlock(128, 256)
        self.down4 = UNetResBlock(256, 512)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNetResBlock(512, 1024)

        # Upsampling blocks
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.upblock4 = UNetResBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.upblock3 = UNetResBlock(512, 256)

        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.upblock2 = UNetResBlock(256, 128)

        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.upblock1 = UNetResBlock(128, 64)

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
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        uncond_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # t の次元調整＆dtype修正
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.float()
        c = c.float()

        # ラベル埋め込み
        label_embed = self.label_embedder(c, t, uncond_mask)

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
