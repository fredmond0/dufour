"""pix2pix: U-Net generator + PatchGAN discriminator.

Deliberately a conventional architecture. The interesting part of this project
is the conditioning stack and the data pipeline, not the network -- and pix2pix
is a well-understood baseline that trains in hours on one GPU.
"""
import torch
import torch.nn as nn


def _norm(c):
    return nn.GroupNorm(min(8, c), c)   # batch-size-agnostic; stable on MPS


class Down(nn.Module):
    def __init__(self, i, o, norm=True):
        super().__init__()
        L = [nn.Conv2d(i, o, 4, 2, 1, bias=not norm)]
        if norm:
            L.append(_norm(o))
        L.append(nn.LeakyReLU(0.2, True))
        self.b = nn.Sequential(*L)

    def forward(self, x):
        return self.b(x)


class Up(nn.Module):
    def __init__(self, i, o, drop=False):
        super().__init__()
        L = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), _norm(o)]
        if drop:
            L.append(nn.Dropout(0.5))
        L.append(nn.ReLU(True))
        self.b = nn.Sequential(*L)

    def forward(self, x, skip):
        return torch.cat([self.b(x), skip], 1)


class Generator(nn.Module):
    """256 -> 256 U-Net, 8 down / 8 up, base width 64."""

    def __init__(self, in_ch, out_ch=3, w=64):
        super().__init__()
        self.d1 = Down(in_ch, w, norm=False)   # 128
        self.d2 = Down(w, w * 2)               # 64
        self.d3 = Down(w * 2, w * 4)           # 32
        self.d4 = Down(w * 4, w * 8)           # 16
        self.d5 = Down(w * 8, w * 8)           # 8
        self.d6 = Down(w * 8, w * 8)           # 4
        self.d7 = Down(w * 8, w * 8)           # 2
        self.d8 = Down(w * 8, w * 8, norm=False)  # 1
        self.u1 = Up(w * 8, w * 8, drop=True)
        self.u2 = Up(w * 16, w * 8, drop=True)
        self.u3 = Up(w * 16, w * 8, drop=True)
        self.u4 = Up(w * 16, w * 8)
        self.u5 = Up(w * 16, w * 4)
        self.u6 = Up(w * 8, w * 2)
        self.u7 = Up(w * 4, w)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(w * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u = self.u1(d8, d7); u = self.u2(u, d6); u = self.u3(u, d5)
        u = self.u4(u, d4); u = self.u5(u, d3); u = self.u6(u, d2)
        u = self.u7(u, d1)
        return self.out(u)


class Discriminator(nn.Module):
    """70x70 PatchGAN over (condition, image) pairs."""

    def __init__(self, in_ch, w=64):
        super().__init__()
        self.b = nn.Sequential(
            Down(in_ch + 3, w, norm=False),
            Down(w, w * 2),
            Down(w * 2, w * 4),
            nn.Conv2d(w * 4, w * 8, 4, 1, 1, bias=False), _norm(w * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(w * 8, 1, 4, 1, 1))

    def forward(self, cond, img):
        return self.b(torch.cat([cond, img], 1))
