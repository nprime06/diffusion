import torch

def single_step(model, xt, t, scheduler): # xt: (B, C, H, W), t: (B,)
    # assume model, xt on same device
    beta = scheduler.get_beta(t).to(device=xt.device).view(-1, 1, 1, 1)
    alpha = scheduler.get_alpha(t).to(device=xt.device).view(-1, 1, 1, 1)
    alpha_prod = scheduler.get_alpha_cumprod(t).to(device=xt.device).view(-1, 1, 1, 1)
    std_t = scheduler.get_standardized_time(t).to(device=xt.device)

    with torch.no_grad():
        pred_eps = model(xt, std_t)
    pred_mu = (xt - (beta / torch.sqrt(1.0 - alpha_prod)) * pred_eps) / torch.sqrt(alpha)
    z = torch.randn_like(xt)
    no_noise = (t == 1).to(device=xt.device).view(-1, 1, 1, 1)
    return pred_mu + (1 - no_noise) * torch.sqrt(beta) * z

def sample(model, xT, scheduler): # xT: (B, C, H, W)
    x = xT.clone()
    history = [x]
    for t in range(scheduler.num_steps, 0, -1): # all t is on cpu
        t_batch = torch.full((x.shape[0],), t, dtype=torch.long)
        x = single_step(model, x, t_batch, scheduler)
        history.append(x)
    return torch.stack(history)