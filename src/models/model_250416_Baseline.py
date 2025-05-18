import torch
import torch.nn as nn


class LabelEmbedder(nn.Module):
    def __init__(self, in_features: int = 2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 1)),  # (B, 512) → (B, 512, 1)
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embed(labels)


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
        self.label_embedder = LabelEmbedder(in_features=label_dim)

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

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 入力エアフォイルデータ (B, in_channels, 248)
            c (Tensor): 条件情報（例: Cl） (B, 1)
            t (Tensor): 時間ステップ (B, 1) または (B,) のテンソル
        Returns:
            Tensor: 生成出力 (B, 2, 248)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        labels = torch.cat([c, t], dim=1)  # (B, 2)
        label_embed = self.label_embedder(labels)  # (B, 512, 1)

        # Downsampling
        d1 = self.down1(x)  # (B, 64, 248)
        d2 = self.down2(self.pool(d1))  # (B, 128, 124)
        d3 = self.down3(self.pool(d2))  # (B, 256, 62)
        d4 = self.down4(self.pool(d3))  # (B, 512, 31)

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(d4))  # (B, 1024, 15)

        # Upsampling with skip connections and label conditioning
        u4 = self.upconv4(bottleneck)  # (B, 512, 30)
        u4 = torch.cat([u4, label_embed], dim=2)  # (B, 512, 31)
        u4 = torch.cat([u4, d4], dim=1)  # (B, 1024, 31)
        u4 = self.upblock4(u4)  # (B, 512, 31)

        u3 = self.upconv3(u4)  # (B, 256, 62)
        u3 = torch.cat([u3, d3], dim=1)  # (B, 512, 62)
        u3 = self.upblock3(u3)  # (B, 256, 62)

        u2 = self.upconv2(u3)  # (B, 128, 124)
        u2 = torch.cat([u2, d2], dim=1)  # (B, 256, 124)
        u2 = self.upblock2(u2)  # (B, 128, 124)

        u1 = self.upconv1(u2)  # (B, 64, 248)
        u1 = torch.cat([u1, d1], dim=1)  # (B, 128, 248)
        u1 = self.upblock1(u1)  # (B, 64, 248)

        if self.output_mode in ["conv1x1", "conv3x3"]:
            return self.output_layer(u1)  # (B, 2, 248)
        else:
            B = x.size(0)
            output = u1.view(B, -1)
            output = self.output_layer(output)
            return output.view(B, self.in_channels, -1)
