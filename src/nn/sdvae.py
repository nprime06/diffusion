from __future__ import annotations

import torch
from diffusers import AutoencoderKL

SD_LATENT_SCALE = 0.18215

def get_vae():
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", subfolder=None, local_files_only=False)
    vae.eval()
    vae = torch.compile(vae)
    return vae

@torch.no_grad()
def encode(vae, x, sample=False):
    x = x * 2.0 - 1.0
    posterior = vae.encode(x).latent_dist
    latents = posterior.sample() if sample else posterior.mode()  # (B, 4, H/8, W/8)
    return latents * SD_LATENT_SCALE  # [0,1] -> N(0,1) (map to [-1,1] for VAE)

@torch.no_grad()
def decode(vae, latents):
    x = vae.decode(latents / SD_LATENT_SCALE).sample  # (B, 3, H, W)
    return (x + 1.0) / 2.0  # N(0,1) -> [0,1]