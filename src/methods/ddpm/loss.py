import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def loss(model, scheduler, x0, labels=None, cfg_proportion=0.8): # x0: (B, C, H, W), labels: (B,)
    # assume model, x0 on same device
    # all t is on cpu
    t = torch.randint(1, scheduler.num_steps + 1, (x0.shape[0],))
    alpha_prod = scheduler.get_alpha_cumprod(t).to(device=x0.device).view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = torch.sqrt(alpha_prod) * x0 + torch.sqrt(1.0 - alpha_prod) * eps
    std_t = scheduler.get_standardized_time(t).to(device=x0.device)
    if labels is not None:
        mask = (torch.rand(labels.shape, device=labels.device) < cfg_proportion).to(torch.long)
        labels = (labels + 1) * mask
    with torch.autocast(device_type=x0.device.type, dtype=torch.bfloat16):
        pred_eps = model(xt, std_t, labels)
        loss = F.mse_loss(pred_eps, eps)
    return loss