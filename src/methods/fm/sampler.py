import torch

SD_LATENT_SCALE = 0.18215

def euler_step(x, v, num_steps): # x: (B, C, H, W), v: (B, C, H, W), num_steps: int
    step_size = torch.full((x.shape[0], 1, 1, 1), 1.0 / num_steps, dtype=x.dtype, device=x.device)
    return x + step_size * v

def sample(model, x0, c, cfg_scale, fm_config, vae): # x0: (B, C, H, W), c: (B,), cfg_scale: (B,)
    x = x0.clone()
    history = [x]
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
                x = vae.decode(x / SD_LATENT_SCALE).sample  # (B, 3, H, W)
                x = (x + 1.0) / 2.0

        history.append(x)
    return torch.stack(history)