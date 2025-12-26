import torch
import torch.optim as optim
import os
import time
from methods.ddpm.loss import loss
from methods.ddpm.sampler import sample
from run_io import save_checkpoint, flush_losses, save_samples_gif

def save_logs(run_dir, loss_buffer, step, model, optimizer, scheduler, device, images_mean, images_std, num_samples=16):
    loss_path = os.path.join(run_dir, "metrics", "loss.jsonl")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    samples_dir = os.path.join(run_dir, "samples")

    flush_losses(loss_path, loss_buffer)
    save_checkpoint(checkpoint_dir, step, model, optimizer)

    xT = torch.randn(num_samples, 1, 28, 28, device=device)
    samples = sample(model, xT, scheduler) # history list from xT to x0; (T, N, 1, 28, 28)
    samples = (samples * images_std.to(device).reshape(1, 1, 1, 28, 28)) + images_mean.to(device).reshape(1, 1, 1, 28, 28)
    save_samples_gif(samples_dir, step, samples)

    return

def train_ddpm(model, dataloader, scheduler, train_config, device, images_mean, images_std): # scheduler, dataloader on cpu; model on device
    params_list = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params_list if p.dim() >= 2]
    nodecay_params = [p for p in params_list if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': train_config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optim_groups, lr=train_config.learning_rate, fused=True)

    loss_buffer = []
    step = 0
    while step < train_config.max_steps:
        for images, labels in dataloader:
            t0 = time.time()
            images, labels = images.to(device).reshape(-1, 1, 28, 28), labels.to(device)
            loss_val = loss(model, images, scheduler)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
                t1 = time.time()
                loss_item = float(loss_val.item())
                loss_buffer.append({"step": step, "loss": loss_item, "lr": optimizer.param_groups[0]["lr"], "time": f"{1000 * (t1 - t0):.2f}ms"})

            if step % train_config.checkpoint_every == 0:
                save_logs(train_config.run_dir, loss_buffer, step, model, optimizer, scheduler, device, images_mean, images_std)
                
            step += 1
            if step >= train_config.max_steps: break
    
    save_logs(train_config.run_dir, loss_buffer, step, model, optimizer, scheduler, device, images_mean, images_std)

    return model