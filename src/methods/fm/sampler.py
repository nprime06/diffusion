import torch

def euler_step(x, v, num_steps): # x: (B, C, H, W), v: (B, C, H, W), num_steps: int
    step_size = torch.full((x.shape[0], 1, 1, 1), 1.0 / num_steps, dtype=x.dtype, device=x.device)
    return x + step_size * v

def sample(model, x0, fm_config): # x0: (B, C, H, W)
    x = x0.clone()
    history = [x]
    for t in range(fm_config.num_steps):
        t_batch = torch.full((x.shape[0],), 1.0 * t / fm_config.num_steps, dtype=x.dtype, device=x.device)
        with torch.no_grad():
            v = model(x, t_batch)

        if fm_config.sampler == 'euler':
            x = euler_step(x, v, fm_config.num_steps)
        else:
            raise ValueError(f"Unsupported sampler: {fm_config.sampler}")

        history.append(x)
    return torch.stack(history)