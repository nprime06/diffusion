import json
import os
import math
import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def write_run_yaml(run_dir, run_info, filename="run.yaml"):
    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        f.write(
            yaml.safe_dump(
                run_info,
                sort_keys=False,
                default_flow_style=False,
            )
        )
    return

def flush_losses(loss_path, loss_buffer):
    with open(loss_path, "a") as f:
        for rec in loss_buffer:
            f.write(json.dumps(rec) + "\n")
    loss_buffer.clear()
    return

def save_checkpoint(checkpoint_dir, step, model, optimizer):
    filename = f"step_{step:08d}.pt"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return

def save_samples_gif(samples_dir, step, samples, max_frames=50, frame_duration_s=0.1, linger_final_s=2.0):
    filename = f"step_{int(step):08d}.gif"
    path = os.path.join(samples_dir, filename)

    samples_numpy = samples.detach().float().cpu().numpy()
    T, N, _, _, _ = samples_numpy.shape

    rows = int(math.floor(math.sqrt(N)))
    cols = int(math.ceil(N / rows))

    # Pick up to max_frames timesteps, evenly spaced, always including the final frame.
    K = int(min(max_frames, T))
    if K <= 1:
        frame_idxs = np.array([T - 1], dtype=int)
    else:
        frame_idxs = np.linspace(0, T - 1, num=K, dtype=int)
        frame_idxs[-1] = T - 1

    duration_ms = int(round(frame_duration_s * 1000.0))
    durations_ms = [duration_ms] * K
    durations_ms[-1] = int(round(linger_final_s * 1000.0))

    frames = []
    for t in frame_idxs:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
        axes = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i >= N:
                continue
            x = samples_numpy[t, i, 0, :, :]  # (H,W)
            ax.imshow(x, cmap="gray")

        fig.suptitle("step " + str(t))
        fig.tight_layout()
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(Image.fromarray(img))
        plt.close(fig)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
    )
    return