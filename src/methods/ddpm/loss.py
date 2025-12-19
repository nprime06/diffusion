import torch
import torch.nn.functional as F

def loss(model, x0, scheduler): # x0: (B, C, H, W)
    # assume model, x0 on same device
    t = torch.randint(1, scheduler.num_steps + 1, (x0.shape[0],), device=x0.device)
    alpha_prod = scheduler.get_alpha_cumprod(t).to(device=x0.device).view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = torch.sqrt(alpha_prod) * x0 + torch.sqrt(1.0 - alpha_prod) * eps
    std_t = scheduler.get_standardized_time(t).to(device=x0.device)
    pred_eps = model(xt, std_t)
    loss = F.mse_loss(pred_eps, eps)
    return loss