import torch
import torch.optim as optim
import os
import time
from methods.fm.loss import loss
from methods.fm.sampler import sample
from run_io import save_checkpoint, flush_losses, save_samples_gif

def save_logs(run_dir, loss_buffer, step, model, optimizer, fm_config, device, images_mean, images_std, num_samples=100):
    loss_path = os.path.join(run_dir, "metrics", "loss.jsonl")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    samples_dir = os.path.join(run_dir, "samples")

    flush_losses(loss_path, loss_buffer)
    save_checkpoint(checkpoint_dir, step, model, optimizer)

    x0 = torch.randn(num_samples, 1, 28, 28, device=device)
    c = torch.arange(10, device=device, dtype=torch.long).repeat_interleave(10) + 1
    cfg_scale = torch.arange(10, device=device, dtype=torch.float).repeat(10) # samples will have rows c = 0, 1, ..., 9, and cols cfg_scale = 0, 1, ..., 9
    samples = sample(model, x0, c, cfg_scale, fm_config) # history list from x0 to xT; (T, N, 1, 28, 28)
    samples = (samples * images_std.to(device).reshape(1, 1, 1, 28, 28)) + images_mean.to(device).reshape(1, 1, 1, 28, 28)
    save_samples_gif(samples_dir, step, samples)

    return

def train_fm(model, dataloader, fm_config, train_config, device, images_mean, images_std): # dataloader on cpu; model on device
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
            loss_val = loss(model, images, labels, train_config.cfg_proportion)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            loss_item = float(loss_val.item())
            loss_buffer.append({"step": step, "loss": loss_item, "lr": optimizer.param_groups[0]["lr"], "time": f"{1000 * (t1 - t0):.2f}ms"})

            if step % train_config.early_checkpoint_every == 0 and step < train_config.num_early_checkpoints * train_config.early_checkpoint_every:
                save_logs(train_config.run_dir, loss_buffer, step, model, optimizer, fm_config, device, images_mean, images_std)
            elif step % train_config.late_checkpoint_every == 0:
                save_logs(train_config.run_dir, loss_buffer, step, model, optimizer, fm_config, device, images_mean, images_std)
                
            step += 1
            if step >= train_config.max_steps: break
    
    save_logs(train_config.run_dir, loss_buffer, step, model, optimizer, fm_config, device, images_mean, images_std)

    return model