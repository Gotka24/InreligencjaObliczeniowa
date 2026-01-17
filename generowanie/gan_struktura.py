import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)


# GENERATOR
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_channels=3, base_ch=64, emb_dim=32):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, emb_dim)

        self.fc = nn.ConvTranspose2d(
            z_dim + emb_dim, base_ch * 8, 4, 1, 0, bias=False
        )  # 4x4

        self.net = nn.Sequential(
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(True),

            self._block(base_ch * 8, base_ch * 4),  # 8x8
            self._block(base_ch * 4, base_ch * 2),  # 16x16
            self._block(base_ch * 2, base_ch),      # 32x32

            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_ch, img_channels, 3, 1, 1),
            nn.Tanh()                               # 64x64
        )

        self.apply(weights_init)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, emb], dim=1)
        x = self.fc(x)
        return self.net(x)


# DISCRIMINATOR (PROJECTION)
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels=3, base_ch=64):
        super().__init__()

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = spectral_norm(nn.Linear(base_ch * 8, 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, base_ch * 8))

        self.apply(weights_init)

    def forward(self, x, labels):
        h = self.conv(x)
        h = torch.sum(h, dim=[2, 3])

        out = self.linear(h).squeeze(1)
        proj = torch.sum(self.embed(labels) * h, dim=1)

        return out + proj
