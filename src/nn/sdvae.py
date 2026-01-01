from __future__ import annotations

import torch
from diffusers import AutoencoderKL

SD_LATENT_SCALE = 0.18215

def get_vae(device=None):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", subfolder=None, local_files_only=False)
    if device is not None:
        vae.to(device)
    vae.eval()
    return vae