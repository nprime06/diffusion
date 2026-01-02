import torch

SD_LATENT_SCALE = 0.18215

def euler_step(x, v, num_steps): # x: (B, C, H, W), v: (B, C, H, W), num_steps: int
    step_size = torch.full((x.shape[0], 1, 1, 1), 1.0 / num_steps, dtype=x.dtype, device=x.device)
    return x + step_size * v

def sample(model, x0, c, cfg_scale, fm_config, image_shape, vae): # x0: (B, C, H, W), c: (B,), cfg_scale: (B,)
    def _decode_latents(x_latents):
        x_img = vae.decode(x_latents / SD_LATENT_SCALE).sample
        x_img = (x_img + 1.0) / 2.0
        return x_img

    x = x0.clone()
    history = torch.empty((fm_config.num_steps, x0.shape[0], *image_shape), device=x.device, dtype=x.dtype)
    if vae is not None:
        with torch.no_grad():
            history[0] = _decode_latents(x)
    else:
        history[0] = x
    
    for t in range(fm_config.num_steps):
        t_batch = torch.full((x.shape[0],), 1.0 * t / fm_config.num_steps, dtype=x.dtype, device=x.device)
        with torch.no_grad():
            pred_v_cond = model(x, t_batch, c)
            pred_v_uncond = model(x, t_batch, None)
            pred_v = pred_v_uncond + cfg_scale.view(-1, 1, 1, 1) * (pred_v_cond - pred_v_uncond)

        if fm_config.sampler == 'euler':
            x = euler_step(x, pred_v, fm_config.num_steps)
        else:
            raise ValueError(f"Unsupported sampler: {fm_config.sampler}")

        if vae is not None:
            with torch.no_grad():
                history[t+1] = _decode_latents(x)
        else:
            history[t+1] = x
    return history