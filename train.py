import argparse
import math

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from tqdm import tqdm

from model import LVPGA


def log_det_jacobian(model, z, log_sigma_sq, dim_z):
    z = z.detach()
    batch = z.shape[0]

    delta = torch.randn(batch, dim_z, device=z.device) * (
        torch.exp(0.5 * log_sigma_sq) + 1e-2
    )
    eps = delta.norm(dim=1, keepdim=True).detach()

    out_z_delta = model(z + delta, mode='dec_enc', grad_enc=False)
    out_z = model(z, mode='dec_enc', grad_enc=False)

    result = dim_z / 2 * torch.log((((out_z_delta - out_z) / eps) ** 2).sum(1))

    return result


def lpvga_loss(model, target, dim_z):
    batch = target.shape[0]
    out, mu, sigma = model(target, mode='enc_dec')

    recon_loss = F.mse_loss(out, target)

    z_target = torch.randn(batch, dim_z, device=target.device)
    z_recon = model(z_target, mode='dec_enc', detach=True)
    z_recon_loss = F.mse_loss(z_recon, z_target)

    z_enc_recon = model(mu.detach(), mode='dec_enc', detach=True)
    z_enc_recon_loss = F.mse_loss(z_enc_recon, mu.detach())

    log_det_loss = log_det_jacobian(model, mu, sigma, dim_z).mean() / dim_z
    kl_loss = -(0.5 * (1 + sigma - mu ** 2 - torch.exp(sigma))).mean()
    nll = (-math.log(2 * math.pi) / 2 - (mu ** 2) / 2).sum(1)
    nll_loss = -nll.mean() / dim_z

    return recon_loss, z_recon_loss, z_enc_recon_loss, log_det_loss, kl_loss, nll_loss


def train(epoch, args, loader, model, optimizer, device):
    pbar = tqdm(loader)

    for i, (img, _) in enumerate(pbar):
        model.zero_grad()

        img = img.to(device)
        recon_loss, z_recon_loss, z_enc_recon_loss, log_det_loss, kl_loss, nll_loss = lpvga_loss(
            model, img, args.dim_z
        )

        if epoch < args.init:
            z_recon_weight = 0
            z_enc_recon_weight = 0
            log_det_weight = 0
            kl_weight = 0
            nll_weight = 0

        else:
            z_recon_weight = args.z_rec
            z_enc_recon_weight = args.z_enc_rec
            log_det_weight = args.log_det
            kl_weight = args.kl
            nll_weight = args.nll

        loss = (
            recon_loss
            + z_recon_weight * z_recon_loss
            + z_enc_recon_weight * z_enc_recon_loss
            + log_det_weight * log_det_loss
            + kl_weight * kl_loss
            + nll_weight * nll_loss
        )
        loss.backward()
        optimizer.step()

        pbar.set_description(
            (
                f'epoch: {epoch + 1}; rec: {recon_loss.item():.3f}; z: {z_recon_loss.item():.3f}; '
                f'z enc: {z_enc_recon_loss.item():.3f}; log det: {log_det_loss.item():.3f}; '
                f'kl: {kl_loss.item():.3f}; nll: {nll_loss.item():.3f}'
            )
        )

        if i % 500 == 0:
            model.eval()

            with torch.no_grad():
                recon, _, _ = model(img)
                sample = model(
                    torch.randn(args.n_sample, args.dim_z, device=device), mode='dec'
                )

            samples = torch.cat(
                [img[: args.n_sample], recon[: args.n_sample], sample[: args.n_sample]],
                0,
            )

            utils.save_image(
                samples,
                f'sample/{str(epoch + 1).zfill(3)}_{str(i).zfill(5)}.png',
                nrow=args.n_sample,
                normalize=True,
                range=(-1, 1),
            )

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--dim_z', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.4)
    parser.add_argument('--init', type=int, default=3)
    parser.add_argument('--z_rec', type=float, default=3e-2)
    parser.add_argument('--z_enc_rec', type=float, default=1e-2)
    parser.add_argument('--log_det', type=float, default=1e-2)
    parser.add_argument('--kl', type=float, default=5e-3)
    parser.add_argument('--nll', type=float, default=1e-2)
    parser.add_argument('--n_sample', type=int, default=20)
    parser.add_argument('path', type=str)

    device = 'cuda'

    args = parser.parse_args()
    print(args)

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    model = LVPGA(args.dim_z, args.dim, args.size)
    print(model)
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.epoch):
        train(i, args, loader, model, optimizer, device)
        torch.save(
            {'model': model.state_dict(), 'args': args}, f'checkpoint/model_{i + 1}.pt'
        )
