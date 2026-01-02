import torch
import torch.optim as optim
import os
import time
import math
from methods.fm.loss import loss
from methods.fm.sampler import sample
from run_io import save_checkpoint, flush_losses, save_samples_gif

SD_LATENT_SCALE = 0.18215

def save_logs(run_dir, loss_buffer, step, model, optimizer, fm_config, device, image_shape, latent_shape, images_mean, images_std, vae, num_samples=9, checkpoint=True):
    loss_path = os.path.join(run_dir, "metrics", "loss.jsonl")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    samples_dir = os.path.join(run_dir, "samples")

    flush_losses(loss_path, loss_buffer)
    if checkpoint:
        save_checkpoint(checkpoint_dir, step, model, optimizer)

    x0 = torch.randn(num_samples, *latent_shape, device=device)
    # c = torch.arange(10, device=device, dtype=torch.long).repeat_interleave(10) + 1
    # cfg_scale = torch.arange(10, device=device, dtype=torch.float).repeat(10) # samples will have rows c = 0, 1, ..., 9, and cols cfg_scale = 0, 1, ..., 9
    c = torch.zeros(num_samples, device=device, dtype=torch.long)
    cfg_scale = torch.zeros(num_samples, device=device, dtype=torch.float)
    samples = sample(model, x0, c, cfg_scale, fm_config, vae) # history list from x0 to xT; (T, N, C, H, W)
    samples = (samples * images_std.to(device).reshape(1, 1, *image_shape)) + images_mean.to(device).reshape(1, 1, *image_shape)
    save_samples_gif(samples_dir, step, samples)

    return

def train_fm(model, dataloader, fm_config, train_config, device, image_shape, latent_shape, images_mean, images_std, vae): # dataloader on cpu; model on device
    params_list = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params_list if p.dim() >= 2]
    nodecay_params = [p for p in params_list if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': train_config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optim_groups, lr=train_config.learning_rate, fused=True)

    warmup_steps = int(0.05 * train_config.max_steps)
    def lr_lambda(current_step):
        if current_step < warmup_steps: return float(current_step + 1) / float(warmup_steps) # linear warmup
        progress = float(current_step - warmup_steps) / float(train_config.max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress)) # cosine decay

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.step()

    loss_buffer = []
    step = 0
    while step < train_config.max_steps:
        for images, labels in dataloader:
            t0 = time.time()
            images, labels = images.to(device).reshape(-1, *image_shape), labels.to(device)
            
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"step {step} time: {1000 * (time.time() - t0):.2f}ms")

            loss_val = loss(model, images, labels, train_config.cfg_proportion)

            optimizer.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            loss_item = float(loss_val.item())
            loss_buffer.append({"step": step, "loss": loss_item, "lr": optimizer.param_groups[0]["lr"], "time": f"{1000 * (t1 - t0):.2f}ms"})

            if step % train_config.early_checkpoint_every == 0 and step < train_config.num_early_checkpoints * train_config.early_checkpoint_every:
                save_logs(train_config.run_dir, loss_buffer, step, model, optimizer, fm_config, device, image_shape, latent_shape, images_mean, images_std, vae, checkpoint=False)
            elif step % train_config.late_checkpoint_every == 0 and step > 0:
                save_logs(train_config.run_dir, loss_buffer, step, model, optimizer, fm_config, device, image_shape, latent_shape, images_mean, images_std, vae)
            
            step += 1
            if step >= train_config.max_steps: break
    
    save_logs(train_config.run_dir, loss_buffer, step, model, optimizer, fm_config, device, image_shape, latent_shape, images_mean, images_std, vae)

    return model