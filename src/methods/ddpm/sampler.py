import torch

SD_LATENT_SCALE = 0.18215

def single_step(model, xt, t, c, cfg_scale, scheduler): # xt: (B, C, H, W), t: (B,), c: (B,), cfg_scale: (B,)
    # assume model, xt on same device
    beta = scheduler.get_beta(t).to(device=xt.device).view(-1, 1, 1, 1)
    alpha = scheduler.get_alpha(t).to(device=xt.device).view(-1, 1, 1, 1)
    alpha_prod = scheduler.get_alpha_cumprod(t).to(device=xt.device).view(-1, 1, 1, 1)
    std_t = scheduler.get_standardized_time(t).to(device=xt.device)

    with torch.no_grad():
        pred_eps_cond = model(xt, std_t, c) # (B, C, H, W)
        pred_eps_uncond = model(xt, std_t, None)
        pred_eps = pred_eps_uncond + cfg_scale.view(-1, 1, 1, 1) * (pred_eps_cond - pred_eps_uncond)
    pred_mu = (xt - (beta / torch.sqrt(1.0 - alpha_prod)) * pred_eps) / torch.sqrt(alpha)
    z = torch.randn_like(xt)
    add_noise = (t != 1).to(device=xt.device, dtype=xt.dtype).view(-1, 1, 1, 1)
    return pred_mu + add_noise * torch.sqrt(beta) * z

def sample(model, xT, c, cfg_scale, scheduler, vae): # xT: (B, C, H, W), c: (B,)
    x = xT.clone()
    history = [x]
    for t in range(scheduler.num_steps, 0, -1): # all t is on cpu
        t_batch = torch.full((x.shape[0],), t, dtype=torch.long)
        x = single_step(model, x, t_batch, c, cfg_scale, scheduler) # c is (B,), dtype=torch.long, on same device as xT

        if vae is not None:
            with torch.no_grad():
                x = vae.decode(x / SD_LATENT_SCALE).sample  # (B, 3, H, W)
                x = (x + 1.0) / 2.0

        history.append(x)
    return torch.stack(history)