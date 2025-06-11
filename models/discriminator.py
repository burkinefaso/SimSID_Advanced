

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ PatchGAN + SpectralNorm Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_filters=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()

        layers = [
            nn.utils.spectral_norm(nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(base_filters * nf_mult_prev, base_filters * nf_mult,
                              kernel_size=4, stride=2, padding=1)
                ),
                nn.BatchNorm2d(base_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers += [
            nn.utils.spectral_norm(
                nn.Conv2d(base_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (B, 1, H/2^n, W/2^n)


# 🧱 Mevcut Basic Discriminator (SimpleDiscriminator)
class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels=1, num_layers=4, base_channels=64):
        super(SimpleDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for i in range(1, num_layers):
            layers += [
                nn.Conv2d(base_channels, base_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(base_channels * 4 * 4, 1)  # assuming input size is 64x64

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# 🛠️ build_disc selector

def build_disc(cfg):
    if hasattr(cfg, 'discriminator_type') and cfg.discriminator_type == "patch":
        return PatchDiscriminator(in_channels=1)
    else:
        return SimpleDiscriminator(in_channels=1)

