import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvRelu(skip_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EfficientUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientUNet, self).__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b4') if pretrained else EfficientNet.from_name('efficientnet-b4')

        self.enc0 = nn.Identity()  # Dummy for skip consistency
        self.enc1 = self.encoder._blocks[0]           # 24x down
        self.enc2 = self.encoder._blocks[1:3]         # 24x
        self.enc3 = self.encoder._blocks[3:6]         # 48x
        self.enc4 = self.encoder._blocks[6:10]        # 64x
        self.enc5 = self.encoder._blocks[10:]         # 160x

        self.mid_conv = ConvRelu(1792, 512)

        self.up4 = UpBlock(512, 448, 256)   # EfficientNet B4 channels
        self.up3 = UpBlock(256, 160, 128)
        self.up2 = UpBlock(128, 56, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def extract_features(self, x):
        endpoints = {}
        x = self.encoder._conv_stem(x)
        x = self.encoder._bn0(x)
        x = self.encoder._swish(x)

        skips = []

        for idx, block in enumerate(self.encoder._blocks):
            drop_connect_rate = self.encoder._global_params.drop_connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate if drop_connect_rate else 0.0)
            if idx in {2, 4, 9, 16}:  # Manually picking layers as per EfficientNet B4 config
                skips.append(x)

        return x, skips[::-1]  # Reverse for decoder use

    def forward(self, x):
        x, skips = self.extract_features(x)
        x = self.mid_conv(x)
        x = self.up4(x, skips[0])
        x = self.up3(x, skips[1])
        x = self.up2(x, skips[2])
        x = self.up1(x, skips[3])
        x = self.up0(x)
        return self.final(x)
