import torch
import torch.nn.functional as F

def loss(model, x1): # x1: (B, C, H, W)
    # assume model, x1 on same device
    # all t is on cpu
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype).view(-1, 1, 1, 1)
    xt = (1 - t) * x0 + t * x1
    v_target = x1 - x0
    v_pred = model(xt, t.view(-1))
    return F.mse_loss(v_pred, v_target)