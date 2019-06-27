import math

from torch import nn
from torch.nn import functional as F


class Conv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, bn=True, activate='relu'
    ):
        super().__init__()

        bias = not bn

        conv = [
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            )
        ]

        if bn:
            conv.append(nn.BatchNorm2d(out_channel))

        if activate is not None:
            if activate == 'relu':
                conv.append(nn.ReLU())

            elif activate == 'lrelu':
                conv.append(nn.LeakyReLU(0.2))

            elif activate == 'tanh':
                conv.append(nn.Tanh())

        self.conv = nn.Sequential(*conv)

    def forward(self, input):
        return self.conv(input)


class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, bn=True, activate='relu'
    ):
        super().__init__()

        bias = not bn

        conv = [
            nn.ConvTranspose2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2 - 1,
                bias=bias,
            )
        ]

        if bn:
            conv.append(nn.BatchNorm2d(out_channel))

        if activate is not None:
            if activate == 'relu':
                conv.append(nn.ReLU())

            elif activate == 'lrelu':
                conv.append(nn.LeakyReLU(0.2))

            elif activate == 'tanh':
                conv.append(nn.Tanh())

        self.conv = nn.Sequential(*conv)

    def forward(self, input):
        return self.conv(input)


class Encoder(nn.Module):
    def __init__(self, dim_z, dim, img_size=64):
        super().__init__()

        self.conv = nn.Sequential(
            Conv2d(3, dim, 5, 2, bn=False, activate='lrelu'),
            Conv2d(dim, dim * 2, 5, 2, bn=False, activate='lrelu'),
            Conv2d(dim * 2, dim * 4, 5, 2, activate='lrelu'),
            Conv2d(dim * 4, dim * 8, 5, 2, activate='lrelu'),
        )

        out_size = img_size // (2 ** 4)

        self.mu = nn.Linear((out_size ** 2) * dim * 8, dim_z)
        self.sigma = nn.Linear((out_size ** 2) * dim * 8, dim_z)
        self.sigma.bias.data.fill_(2 * math.log(0.1))

    def forward(self, input):
        batch = input.shape[0]
        out = self.conv(input)
        out = out.view(batch, -1)
        mu = self.mu(out)
        sigma = self.sigma(out)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, dim_z, dim, img_size=64):
        super().__init__()

        self.dim = dim
        self.out_size = img_size // (2 ** 4)

        self.mu = nn.Sequential(
            nn.Linear(dim_z, (self.out_size ** 2) * dim * 8), nn.ReLU()
        )

        self.conv = nn.Sequential(
            ConvTranspose2d(dim * 8, dim * 4, 6, 2),
            ConvTranspose2d(dim * 4, dim * 2, 6, 2),
            ConvTranspose2d(dim * 2, dim, 6, 2, bn=False),
            ConvTranspose2d(dim, 3, 6, 2, bn=False, activate='tanh')
        )

    def forward(self, input):
        batch = input.shape[0]
        out = self.mu(input)
        out = out.view(batch, self.dim * 8, self.out_size, self.out_size)
        out = self.conv(out)

        return out


class LVPGA(nn.Module):
    def __init__(self, dim_z, dim=64, img_size=64):
        super().__init__()

        self.enc = Encoder(dim_z, dim, img_size)
        self.dec = Decoder(dim_z, dim, img_size)

    def enc_dec(self, input):
        mu, sigma = self.enc(input)
        recon = self.dec(mu)

        return recon, mu, sigma

    def dec_enc(self, input, detach=False, grad_enc=True):
        recon = self.dec(input)

        if detach:
            recon = recon.detach()

        if grad_enc:
            mu, _ = self.enc(recon)

        else:
            mu, _ = self.enc(recon)
            mu_detach, _ = self.enc(recon.detach())
            mu = mu + mu.detach() - mu_detach

        return mu

    def forward(self, input, mode='enc_dec', detach=False, grad_enc=True):
        if mode == 'enc_dec':
            return self.enc_dec(input)

        elif mode == 'dec_enc':
            return self.dec_enc(input, detach, grad_enc)

        elif mode == 'dec':
            return self.dec(input)
