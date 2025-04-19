import torch
import torch.nn as nn
from numpy import sqrt

""" Legacy Network Definitions
class Lin_Conv_Res_Block(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        
        self.cnvlv_stack = nn.Sequential(
            nn.Conv1d(1, 50, 7, padding=3, padding_mode='circular'),
            nn.LeakyReLU(),
            nn.Conv1d(50, 25, 5, padding=2, padding_mode='circular'),
            nn.LeakyReLU(),
            nn.Conv1d(25, 1, 3, padding=1, padding_mode='circular'),
            #nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2* n_feat + 2, n_feat + 2),
            nn.LeakyReLU(),
            nn.Linear(n_feat + 2, n_feat),
            nn.LeakyReLU()
        )
    

    def forward(self, A, B, c, t):
        Bout = self.cnvlv_stack(torch.cat((A, B), dim=1).unsqueeze(1)).squeeze(1)
        Aout = self.linear_relu_stack(torch.cat((Bout, A, c, t), dim=1))
        return Aout, Bout
    
class Lin_Conv_Network_3(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        
        self.linear_relu_stack_1 = nn.Sequential(
            nn.Linear(n_feat + 2, n_feat + 2),
            nn.LeakyReLU(),
            nn.Linear(n_feat + 2, n_feat),
            nn.LeakyReLU()
        )
        
        self.Lin_Conv_Block_1 = Lin_Conv_Res_Block(n_feat)

        self.linear_relu_stack_2 = nn.Sequential(
            nn.Linear(2 * n_feat + 2, n_feat + 2),
            nn.LeakyReLU(),
            nn.Linear(n_feat + 2, n_feat),
        )
    
    def forward(self, x, c, t):
        Aout1 = self.linear_relu_stack_1(torch.cat((x, c, t), dim=1))

        Aout2, Bout2 = self.Lin_Conv_Block_1(x, Aout1, c, t)
        # Aout2, Bout2 = self.Lin_Conv_Block_1(Aout1, x, c, t)
        
        Aout3 = self.linear_relu_stack_2(torch.cat((Aout2, Bout2, c, t), dim=1))
        return Aout3
    
class Lin_Conv_Network_4(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        
        self.linear_relu_stack_1 = nn.Sequential(
            nn.Linear(n_feat + 2, n_feat + 2),
            nn.LeakyReLU(),
            nn.Linear(n_feat + 2, n_feat),
            nn.LeakyReLU()
        )
        
        self.Lin_Conv_Block_1 = Lin_Conv_Res_Block(n_feat)
        self.Lin_Conv_Block_2 = Lin_Conv_Res_Block(n_feat)

        self.linear_relu_stack_2 = nn.Sequential(
            nn.Linear(2 * n_feat + 2, n_feat + 2),
            nn.LeakyReLU(),
            nn.Linear(n_feat + 2, n_feat),
        )
    
    def forward(self, x, c, t):
        Aout1 = self.linear_relu_stack_1(torch.cat((x, c, t), dim=1))

        Aout2, Bout2 = self.Lin_Conv_Block_1(x, Aout1, c, t)
        Aout3, Bout3 = self.Lin_Conv_Block_2(Bout2, Aout2, c, t)

        # Aout2, Bout2 = self.Lin_Conv_Block_1(x, Aout1, c, t)
        # Aout3, Bout3 = self.Lin_Conv_Block_2(Bout2, Aout2 + x, c, t)

        # Aout2, Bout2 = self.Lin_Conv_Block_1(Aout1, x, c, t)
        # Aout3, Bout3 = self.Lin_Conv_Block_2(Aout2, Bout2, c, t)

        Aout4 = self.linear_relu_stack_2(torch.cat((Aout3, Bout3, c, t), dim=1))
        return Aout4
"""


# --------------------------------------------------------------------------- #
# Fully Connected Feed Forward Networks
# --------------------------------------------------------------------------- #
class FC_FF(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.linear_relu = nn.Sequential(
            nn.Linear(n_feat + 2, n_feat), nn.LeakyReLU(), nn.Linear(n_feat, n_feat), nn.Unflatten(1, (2, 248))
        )

    def forward(self, x, c, t):
        # concatenate x, c, t and pass through the network
        logits = x.flatten(1)
        logits = torch.cat((logits, c, t), dim=1)
        logits = self.linear_relu(logits)
        return logits


class FC_FF_V2(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.linear_relu = nn.Sequential(
            nn.Linear(n_feat + 2, n_feat + 2),
            nn.LeakyReLU(),
            nn.Linear(n_feat + 2, n_feat),
            nn.LeakyReLU(),
            nn.Linear(n_feat, n_feat),
            nn.Unflatten(1, (2, 248)),
        )

    def forward(self, x, c, t):
        # concatenate x, c, t and pass through the network
        logits = x.flatten(1)
        logits = torch.cat((logits, c, t), dim=1)
        logits = self.linear_relu(logits)
        return logits


# --------------------------------------------------------------------------- #
# Convolutional Networks
# --------------------------------------------------------------------------- #
class Conv_Network_1(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 5, padding=2, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 + 64 * n_feat // 2, 16 * n_feat),
            nn.LeakyReLU(),
            nn.Linear(16 * n_feat, n_feat),
            nn.Unflatten(1, (2, 248)),
        )

    def forward(self, x, c, t):
        logits = self.conv_stack(x)
        logits = self.linear_relu_stack(torch.cat((logits.flatten(1), c, t), dim=1))
        return logits


class Lin_Conv_Network_1(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_feat + 2, n_feat + 2), nn.LeakyReLU(), nn.Linear(n_feat + 2, n_feat), nn.LeakyReLU()
        )

        self.cnvlv = nn.Sequential(
            nn.Conv1d(2, 48, 7, padding=3, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(48, 24, 5, padding=2, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(24, 2, 3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
        )

        self.linear_relu_stack_2 = nn.Sequential(
            nn.Linear(2 * n_feat + 2, n_feat + 2),
            nn.LeakyReLU(),
            nn.Linear(n_feat + 2, n_feat),
        )

    def forward(self, x, c, t):
        # concatenate x, c, t and pass through the network
        # print(x.shape, c.shape, t.shape)
        x = x.flatten(1)
        logits = self.linear_relu_stack(torch.cat((x, c, t), dim=1))
        logits = logits.unflatten(1, (2, 248))
        logits = self.cnvlv(logits).flatten(1)
        logits = self.linear_relu_stack_2(torch.cat((x, logits, c, t), dim=1))
        logits = logits.unflatten(1, (2, 248))
        return logits


class Lin_Conv_Network_2(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        cdim_1 = 50  # 64; 32; 50
        cdim_2 = 25  # 16, 32; 25
        grps = 1  # 1

        cnvlv_layers = [
            nn.Conv1d(4, cdim_1, 7, padding=3, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(cdim_1, cdim_2, 5, padding=2, groups=grps, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(cdim_2, 2, 3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
        ]

        self.cnvlv = nn.Sequential(*cnvlv_layers)
        self.cnvlv2 = nn.Sequential(*cnvlv_layers)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_feat + 2, n_feat + 2), nn.LeakyReLU(), nn.Linear(n_feat + 2, n_feat), nn.LeakyReLU()
        )

        self.linear_relu_stack_2 = nn.Sequential(
            nn.Linear(2 * n_feat + 2, n_feat + 2), nn.LeakyReLU(), nn.Linear(n_feat + 2, n_feat), nn.LeakyReLU()
        )

        self.linear_relu_stack_3 = nn.Sequential(
            nn.Linear(3 * n_feat + 2, n_feat + 2),
            nn.LeakyReLU(),
            nn.Linear(n_feat + 2, n_feat),
        )

    def forward(self, x, c, t):
        logits1 = self.linear_relu_stack(torch.cat((x.flatten(1), c, t), dim=1)).unflatten(1, (2, 248))
        logits2 = self.cnvlv(torch.cat((x, logits1), dim=1))
        logits1 = self.linear_relu_stack_2(torch.cat((logits1.flatten(1), logits2.flatten(1), c, t), dim=1)).unflatten(
            1, (2, 248)
        )
        logits2 = self.cnvlv2(torch.cat((logits1, logits2), dim=1))
        logits1 = self.linear_relu_stack_3(
            torch.cat((x.flatten(1), logits2.flatten(1), logits1.flatten(1), c, t), dim=1)
        ).unflatten(1, (2, 248))
        return logits1


# --------------------------------------------------------------------------- #
# Mixing type Networks
# --------------------------------------------------------------------------- #
class Mix_Conv_Network_1(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        layers = [
            nn.Conv1d(2, 64, 7, padding=3, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 5, padding=2, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(),
            nn.Flatten(1),  # Flatten only before linear layers?
        ]

        self.conv_1 = nn.Sequential(*layers)
        self.conv_2 = nn.Sequential(*layers)
        self.conv_3 = nn.Sequential(*layers)

        self.linear_relu_mixer = nn.Sequential(
            nn.Linear(3 * 64 * n_feat // 2 + 2, 14 * n_feat),
            nn.LeakyReLU(),
            nn.Linear(14 * n_feat, n_feat),
            nn.Unflatten(1, (2, 248)),
        )

    def forward(self, x, c, t):
        logits1 = self.conv_1(x)
        logits2 = self.conv_2(x)
        logits3 = self.conv_3(x)

        mix = torch.cat((logits1, logits2, logits3, c, t), dim=1)
        mix = self.linear_relu_mixer(mix)
        return mix


# --------------------------------------------------------------------------- #
# U-Net Networks
# --------------------------------------------------------------------------- #
class UNet_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(13, 7, 3), dilation=(1, 1, 1)):
        super(UNet_conv, self).__init__()
        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size[0],
                dilation=dilation[0],
                padding=dilation[0] * (kernel_size[0] - 1) // 2,
                padding_mode="circular",
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size[1],
                dilation=dilation[1],
                padding=dilation[1] * (kernel_size[1] - 1) // 2,
                padding_mode="circular",
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size[2],
                dilation=dilation[2],
                padding=dilation[2] * (kernel_size[2] - 1) // 2,
                padding_mode="circular",
            ),
            nn.LeakyReLU(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, in_channels=2, n_feat=0, n_channel=64):

        super(UNet, self).__init__()

        self.down1_1 = UNet_conv(in_channels, n_channel)
        self.down1_2 = nn.MaxPool1d(2)

        self.down2_1 = UNet_conv(n_channel, 2 * n_channel)
        self.down2_2 = nn.MaxPool1d(2)

        self.down3_1 = UNet_conv(2 * n_channel, 4 * n_channel)  # , dilation=(1,1,1))
        self.down3_2 = nn.MaxPool1d(2)

        self.down4_1 = UNet_conv(4 * n_channel, 8 * n_channel, dilation=(1, 1, 1))
        self.down4_2 = nn.MaxPool1d(2)

        self.latent = UNet_conv(8 * n_channel, 16 * n_channel, dilation=(1, 1, 1))
        # self.latent = UNet_conv(4*n_channel, 8*n_channel)

        self.up1_1 = nn.ConvTranspose1d(16 * n_channel, 8 * n_channel, 2, stride=2)
        self.up1_2 = UNet_conv(16 * n_channel, 8 * n_channel, dilation=(1, 1, 1))

        self.up2_1 = nn.ConvTranspose1d(8 * n_channel, 4 * n_channel, 2, stride=2)
        self.up2_2 = UNet_conv(8 * n_channel, 4 * n_channel)  # , dilation=(1,1,1))

        self.up3_1 = nn.ConvTranspose1d(4 * n_channel, 2 * n_channel, 2, stride=2)
        self.up3_2 = UNet_conv(4 * n_channel, 2 * n_channel)

        self.up4_1 = nn.ConvTranspose1d(2 * n_channel, n_channel, 2, stride=2)
        self.up4_2 = UNet_conv(2 * n_channel, n_channel)

        # Old names, prior 28/5: V1 3x3 conv, V2 FC + LeakyReLU + FC, V3 1x1 conv, V4 FC
        # out = nxn conv
        # self.out = nn.Conv1d(n_channel, in_channels, 3, padding=1, padding_mode='circular')
        self.out = nn.Conv1d(n_channel, in_channels, kernel_size=1)
        # self.out = nn.Conv1d(n_channel, in_channels, kernel_size=1, groups=2)

        # out = FC
        if False:
            self.out = nn.Sequential(
                nn.Flatten(1), nn.Linear(n_channel * n_feat // 2, n_feat), nn.Unflatten(1, (2, 248))
            )

        # out = FC NN = FC + LeakyReLU + FC
        # For homogenous reduction of feature space in a two step process
        # hom_fac = sqrt(1 / (n_channel//2))

        # self.out = nn.Sequential(
        #     nn.Flatten(1),
        #     nn.Linear(n_channel*n_feat//2, round(n_channel*(n_feat//2)*hom_fac)),
        #     nn.LeakyReLU(),
        #     nn.Linear(round(n_channel*(n_feat//2)*hom_fac), n_feat),
        #     nn.Unflatten(1, (2, 248))
        # )

        # Todo nn.Embedding
        self.embed = nn.Sequential(
            nn.Linear(2, 8 * n_channel),
            nn.LeakyReLU(),
            nn.Linear(8 * n_channel, 8 * n_channel),
            nn.LeakyReLU(),
            nn.Unflatten(1, (8 * n_channel, 1)),
        )

    def forward(self, x, c, t):
        emb = self.embed(torch.cat((c, t), dim=1))  # B x 512 x 1

        down1 = self.down1_1(x)
        down2 = self.down2_1(self.down1_2(down1))
        down3 = self.down3_1(self.down2_2(down2))
        down4 = self.down4_1(self.down3_2(down3))

        out = self.latent(self.down4_2(down4))
        # out = self.latent(self.down3_2(down3))

        out = self.up1_1(out)
        # Embedding after first upsample, in latent space might be better but fits here because skip connections need additional channels for concatenation since downsampling by 2 in each step leads to 248 --> 124 --> 62 --> 31 --> 15 --> 30! cant concatenate 31 and 15 Now 248 --> 124 --> 62 --> 31 --> 15 --> 30 + 1 --> 62 --> 124 --> 248
        out = torch.cat((out, emb), dim=2)
        out = self.up1_2(torch.cat((out, down4), dim=1))

        out = self.up2_1(out)
        out = self.up2_2(torch.cat((out, down3), dim=1))

        out = self.up3_1(out)
        out = self.up3_2(torch.cat((out, down2), dim=1))

        out = self.up4_1(out)
        out = self.up4_2(torch.cat((out, down1), dim=1))

        out = self.out(out)

        return out


class UNet_Res_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(7, 5), dilation=(1, 1)):
        super(UNet_Res_conv, self).__init__()
        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size[0],
                dilation=dilation[0],
                padding=dilation[0] * (kernel_size[0] - 1) // 2,
                padding_mode="circular",
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size[1],
                dilation=dilation[1],
                padding=dilation[1] * (kernel_size[1] - 1) // 2,
                padding_mode="circular",
            ),
        ]

        self.n_channel_fac = out_channels / in_channels

        self.conv = nn.Sequential(*layers)
        self.out = nn.LeakyReLU()

        if self.n_channel_fac < 1:
            # Merge Feature dimension to match number of channels in y
            # Todo use average pooling to reduce feature dimension instead of 1x1 conv
            self.merge = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv(x)

        if self.n_channel_fac > 1:  # Expand x along second axis to match the number of channels in y
            return self.out(x.repeat(1, int(self.n_channel_fac), 1) + y)
        elif self.n_channel_fac < 1:  # Merge x and y by 1x1 conv.
            return self.out(self.merge(x) + y)
        else:
            return self.out(x + y)


# Todo this hardcoding is terrible, make it more flexible via a layer and onv. arguments
class UNet_Res(nn.Module):
    def __init__(self, in_channels=2, n_feat=0, n_channel=64):

        super(UNet_Res, self).__init__()

        self.down1_0 = UNet_Res_conv(in_channels, n_channel)
        self.down1_1 = UNet_Res_conv(n_channel, n_channel)
        # self.down1_2 = UNet_Res_conv(n_channel, n_channel)
        # self.down1_3 = UNet_Res_conv(n_channel, n_channel)
        self.down1_p = nn.MaxPool1d(2)

        self.down2_0 = UNet_Res_conv(n_channel, 2 * n_channel)
        self.down2_1 = UNet_Res_conv(2 * n_channel, 2 * n_channel)
        # self.down2_2 = UNet_Res_conv(2*n_channel, 2*n_channel)
        # self.down2_3 = UNet_Res_conv(2*n_channel, 2*n_channel)
        self.down2_p = nn.MaxPool1d(2)

        self.down3_0 = UNet_Res_conv(2 * n_channel, 4 * n_channel, dilation=(1, 1, 1))
        self.down3_1 = UNet_Res_conv(4 * n_channel, 4 * n_channel, dilation=(1, 1, 1))
        # self.down3_2 = UNet_Res_conv(4*n_channel, 4*n_channel, dilation=(1,1,1))
        # self.down3_3 = UNet_Res_conv(4*n_channel, 4*n_channel, dilation=(1,1,1))
        self.down3_p = nn.MaxPool1d(2)

        self.down4_0 = UNet_Res_conv(4 * n_channel, 8 * n_channel, dilation=(1, 1, 1))
        self.down4_1 = UNet_Res_conv(8 * n_channel, 8 * n_channel, dilation=(1, 1, 1))
        # self.down4_2 = UNet_Res_conv(8*n_channel, 8*n_channel, dilation=(1,1,1))
        # self.down4_3 = UNet_Res_conv(8*n_channel, 8*n_channel, dilation=(1,1,1))
        self.down4_p = nn.MaxPool1d(2)

        self.latent_0 = UNet_Res_conv(8 * n_channel, 16 * n_channel, dilation=(1, 1, 1))
        self.latent_1 = UNet_Res_conv(16 * n_channel, 16 * n_channel, dilation=(1, 1, 1))
        # self.latent_2 = UNet_Res_conv(16*n_channel, 16*n_channel, dilation=(1,1,1))
        # self.latent_3 = UNet_Res_conv(16*n_channel, 16*n_channel, dilation=(1,1,1))

        self.up1_1 = nn.ConvTranspose1d(16 * n_channel, 8 * n_channel, 2, stride=2)
        self.up1_2 = UNet_Res_conv(16 * n_channel, 8 * n_channel, dilation=(1, 1, 1))
        self.up1_3 = UNet_Res_conv(8 * n_channel, 8 * n_channel, dilation=(1, 1, 1))
        # self.up1_4 = UNet_Res_conv(8*n_channel, 8*n_channel, dilation=(1,1,1))
        # self.up1_5 = UNet_Res_conv(8*n_channel, 8*n_channel, dilation=(1,1,1))

        self.up2_1 = nn.ConvTranspose1d(8 * n_channel, 4 * n_channel, 2, stride=2)
        self.up2_2 = UNet_Res_conv(8 * n_channel, 4 * n_channel, dilation=(1, 1, 1))
        self.up2_3 = UNet_Res_conv(4 * n_channel, 4 * n_channel, dilation=(1, 1, 1))
        # self.up2_4 = UNet_Res_conv(4*n_channel, 4*n_channel, dilation=(1,1,1))
        # self.up2_5 = UNet_Res_conv(4*n_channel, 4*n_channel, dilation=(1,1,1))

        self.up3_1 = nn.ConvTranspose1d(4 * n_channel, 2 * n_channel, 2, stride=2)
        self.up3_2 = UNet_Res_conv(4 * n_channel, 2 * n_channel)
        self.up3_3 = UNet_Res_conv(2 * n_channel, 2 * n_channel)
        # self.up3_4 = UNet_Res_conv(2*n_channel, 2*n_channel)
        # self.up3_5 = UNet_Res_conv(2*n_channel, 2*n_channel)

        self.up4_1 = nn.ConvTranspose1d(2 * n_channel, n_channel, 2, stride=2)
        self.up4_2 = UNet_Res_conv(2 * n_channel, n_channel)
        self.up4_3 = UNet_Res_conv(n_channel, n_channel)
        # self.up4_4 = UNet_Res_conv(n_channel, n_channel)
        # self.up4_5 = UNet_Res_conv(n_channel, n_channel)

        # Old names, prior 28/5: V1 3x3 conv, V2 FC + LeakyReLU + FC, V3 1x1 conv, V4 FC
        # out = nxn conv
        # self.out = nn.Conv1d(n_channel, in_channels, 3, padding=1, padding_mode='circular')
        self.out = nn.Conv1d(n_channel, in_channels, kernel_size=1)
        # self.out = nn.Conv1d(n_channel, in_channels, kernel_size=1, groups=2)

        # out = FC
        if False:
            self.out = nn.Sequential(
                nn.Flatten(1), nn.Linear(n_channel * n_feat // 2, n_feat), nn.Unflatten(1, (2, 248))
            )

        # out = FC NN = FC + LeakyReLU + FC
        # For homogenous reduction of feature space in a two step process
        # hom_fac = sqrt(1 / (n_channel//2))

        # self.out = nn.Sequential(
        #     nn.Flatten(1),
        #     nn.Linear(n_channel*n_feat//2, round(n_channel*(n_feat//2)*hom_fac)),
        #     nn.LeakyReLU(),
        #     nn.Linear(round(n_channel*(n_feat//2)*hom_fac), n_feat),
        #     nn.Unflatten(1, (2, 248))
        # )

        # Todo nn.Embedding
        self.embed = nn.Sequential(
            nn.Linear(2, 8 * n_channel),
            nn.LeakyReLU(),
            nn.Linear(8 * n_channel, 8 * n_channel),
            nn.LeakyReLU(),
            nn.Unflatten(1, (8 * n_channel, 1)),
        )

    def forward(self, x, c, t):
        emb = self.embed(torch.cat((c, t), dim=1))  # B x 512 x 1

        down1 = self.down1_0(x)
        down1 = self.down1_1(down1)
        # down1 = self.down1_2(down1)
        # down1 = self.down1_3(down1)

        down2 = self.down2_0(self.down1_p(down1))
        down2 = self.down2_1(down2)
        # down2 = self.down2_2(down2)
        # down2 = self.down2_3(down2)

        down3 = self.down3_0(self.down2_p(down2))
        down3 = self.down3_1(down3)
        # down3 = self.down3_2(down3)
        # down3 = self.down3_3(down3)

        down4 = self.down4_0(self.down3_p(down3))
        down4 = self.down4_1(down4)
        # down4 = self.down4_2(down4)
        # down4 = self.down4_3(down4)

        out = self.latent_0(self.down4_p(down4))
        out = self.latent_1(out)
        # out = self.latent_2(out)
        # out = self.latent_3(out)

        out = self.up1_1(out)
        # Embedding after first upsample, in latent space might be better but fits here because skip connections need additional channels for concatenation since downsampling by 2 in each step leads to 248 --> 124 --> 62 --> 31 --> 15 --> 30! cant concatenate 31 and 15 Now 248 --> 124 --> 62 --> 31 --> 15 --> 30 + 1 --> 62 --> 124 --> 248
        # print(out.shape, emb.shape)
        out = torch.cat((out, emb), dim=2)
        out = self.up1_2(torch.cat((out, down4), dim=1))
        out = self.up1_3(out)
        # out = self.up1_4(out)
        # out = self.up1_5(out)

        out = self.up2_1(out)
        out = self.up2_2(torch.cat((out, down3), dim=1))
        out = self.up2_3(out)
        # out = self.up2_4(out)
        # out = self.up2_5(out)

        out = self.up3_1(out)
        out = self.up3_2(torch.cat((out, down2), dim=1))
        out = self.up3_3(out)
        # out = self.up3_4(out)
        # out = self.up3_5(out)

        out = self.up4_1(out)
        out = self.up4_2(torch.cat((out, down1), dim=1))
        out = self.up4_3(out)
        # out = self.up4_4(out)
        # out = self.up4_5(out)

        out = self.out(out)

        return out


# --------------------------------------------------------------------------- #
# Legacy Networks Definition Unet Res 3x

""" class UNet_Res(nn.Module):
    def __init__(self, in_channels = 2, n_feat = 0, n_channel =32):

        super(UNet_Res, self).__init__()
        
        self.down1_0 = UNet_Res_conv(in_channels, n_channel)
        self.down1_05 = UNet_Res_conv(n_channel, n_channel)
        self.down1_1 = UNet_Res_conv(n_channel, n_channel)
        self.down1_2 = nn.MaxPool1d(2)

        self.down2_0 = UNet_Res_conv(n_channel, 2*n_channel)
        self.down2_05 = UNet_Res_conv(2*n_channel, 2*n_channel)
        self.down2_1 = UNet_Res_conv(2*n_channel, 2*n_channel)
        self.down2_2 = nn.MaxPool1d(2)

        self.down3_0 = UNet_Res_conv(2*n_channel, 4*n_channel, dilation=(1,1,1))
        self.down3_05 = UNet_Res_conv(4*n_channel, 4*n_channel, dilation=(1,1,1))
        self.down3_1 = UNet_Res_conv(4*n_channel, 4*n_channel, dilation=(1,1,1))
        self.down3_2 = nn.MaxPool1d(2)

        self.down4_0 = UNet_Res_conv(4*n_channel, 8*n_channel, dilation=(1,1,1))
        self.down4_05 = UNet_Res_conv(8*n_channel, 8*n_channel, dilation=(1,1,1))
        self.down4_1 = UNet_Res_conv(8*n_channel, 8*n_channel, dilation=(1,1,1))
        self.down4_2 = nn.MaxPool1d(2)


        self.latent_0 = UNet_Res_conv(8*n_channel, 16*n_channel, dilation=(1,1,1))
        self.latent_05 = UNet_Res_conv(16*n_channel, 16*n_channel, dilation=(1,1,1))
        self.latent_1 = UNet_Res_conv(16*n_channel, 16*n_channel, dilation=(1,1,1))


        self.up1_1 = nn.ConvTranspose1d(16*n_channel, 8*n_channel, 2, stride=2)
        self.up1_2 = UNet_Res_conv(16*n_channel, 8*n_channel, dilation=(1,1,1))
        self.up1_3 = UNet_Res_conv(8*n_channel, 8*n_channel, dilation=(1,1,1))
        self.up1_4 = UNet_Res_conv(8*n_channel, 8*n_channel, dilation=(1,1,1))

        self.up2_1 = nn.ConvTranspose1d(8*n_channel, 4*n_channel, 2, stride=2)
        self.up2_2 = UNet_Res_conv(8*n_channel, 4*n_channel, dilation=(1,1,1))
        self.up2_3 = UNet_Res_conv(4*n_channel, 4*n_channel, dilation=(1,1,1))
        self.up2_4 = UNet_Res_conv(4*n_channel, 4*n_channel, dilation=(1,1,1))

        self.up3_1 = nn.ConvTranspose1d(4*n_channel, 2*n_channel, 2, stride=2)
        self.up3_2 = UNet_Res_conv(4*n_channel, 2*n_channel)
        self.up3_3 = UNet_Res_conv(2*n_channel, 2*n_channel)
        self.up3_4 = UNet_Res_conv(2*n_channel, 2*n_channel)

        self.up4_1 = nn.ConvTranspose1d(2*n_channel, n_channel, 2, stride=2)
        self.up4_2 = UNet_Res_conv(2*n_channel, n_channel)
        self.up4_3 = UNet_Res_conv(n_channel, n_channel)
        self.up4_4 = UNet_Res_conv(n_channel, n_channel)
        
        # Old names, prior 28/5: V1 3x3 conv, V2 FC + LeakyReLU + FC, V3 1x1 conv, V4 FC
        #out = nxn conv
        #self.out = nn.Conv1d(n_channel, in_channels, 3, padding=1, padding_mode='circular')
        self.out = nn.Conv1d(n_channel, in_channels, kernel_size=1)
        #self.out = nn.Conv1d(n_channel, in_channels, kernel_size=1, groups=2)

        #out = FC
        if False:
            self.out = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(n_channel*n_feat//2, n_feat),
                nn.Unflatten(1, (2, 248))
            )

        # out = FC NN = FC + LeakyReLU + FC
        # For homogenous reduction of feature space in a two step process
        # hom_fac = sqrt(1 / (n_channel//2))

        # self.out = nn.Sequential(
        #     nn.Flatten(1),
        #     nn.Linear(n_channel*n_feat//2, round(n_channel*(n_feat//2)*hom_fac)),
        #     nn.LeakyReLU(),
        #     nn.Linear(round(n_channel*(n_feat//2)*hom_fac), n_feat),
        #     nn.Unflatten(1, (2, 248))
        # )

        # Todo nn.Embedding
        self.embed = nn.Sequential(
            nn.Linear(2, 8*n_channel),
            nn.LeakyReLU(),
            nn.Linear(8*n_channel, 8*n_channel),
            nn.LeakyReLU(),
            nn.Unflatten(1, (8*n_channel, 1))
        )


    def forward(self, x, c, t):
        emb = self.embed(torch.cat((c, t), dim=1)) # B x 512 x 1

        down1 = self.down1_1(self.down1_05(self.down1_0(x)))
        down2 = self.down2_1(self.down2_05(self.down2_0(self.down1_2(down1))))
        down3 = self.down3_1(self.down3_05(self.down3_0(self.down2_2(down2))))
        down4 = self.down4_1(self.down4_05(self.down4_0(self.down3_2(down3))))

        # down1 = self.down1_1(self.down1_0(x))
        # down2 = self.down2_1(self.down2_0(self.down1_2(down1)))
        # down3 = self.down3_1(self.down3_0(self.down2_2(down2)))
        # down4 = self.down4_1(self.down4_0(self.down3_2(down3)))

        out = self.latent_1(self.latent_05(self.latent_0(self.down4_2(down4))))
        out = self.latent_1(self.latent_0(self.down4_2(down4)))

        out = self.up1_1(out)
        # Embedding after first upsample, in latent space might be better but fits here because skip connections need additional channels for concatenation since downsampling by 2 in each step leads to 248 --> 124 --> 62 --> 31 --> 15 --> 30! cant concatenate 31 and 15 Now 248 --> 124 --> 62 --> 31 --> 15 --> 30 + 1 --> 62 --> 124 --> 248
        out = torch.cat((out, emb), dim=2)
        out = self.up1_2(torch.cat((out, down4), dim=1))
        out = self.up1_3(out)
        out = self.up1_4(out)

        out = self.up2_1(out)
        out = self.up2_2(torch.cat((out, down3), dim=1))
        out = self.up2_3(out)
        out = self.up2_4(out)
        
        out = self.up3_1(out)
        out = self.up3_2(torch.cat((out, down2), dim=1))
        out = self.up3_3(out)
        out = self.up3_4(out)

        out = self.up4_1(out)
        out = self.up4_2(torch.cat((out, down1), dim=1))
        out = self.up4_3(out)
        out = self.up4_4(out)

        out = self.out(out)

        return out """


# --------------------------------------------------------------------------- #
""" Yonekura's Network Definitions
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool1d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module): 
    def __init__(self, in_channels, n_feat = 256):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool1d(31), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(2, 2*n_feat)
        self.contextembed2 = EmbedFC(2, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose1d(2 * n_feat, 2 * n_feat, 31, 31), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv1d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv1d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # for debugging, save the inputs
        # torch.save(x, 'x.pt')
        # torch.save(c, 'c.pt')
        # torch.save(t, 't.pt')
        # torch.save(context_mask, 'context_mask.pt')

        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        context_mask = context_mask.repeat(1,1)
        c = torch.sigmoid(c) * (1 - context_mask) # need to flip 0 <-> 1
        c = torch.cat((c,context_mask),dim=-1)
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out
"""
