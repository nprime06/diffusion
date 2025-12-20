import json
import os
import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = "/Users/william/Desktop/Random/diffusion/logs/ddpm/mnist_ddpm_unet_2025-12-20_17-53-28"

CUTOFF = 50000

steps = []
losses = []

LOSS_DIR = os.path.join(RUN_DIR, "metrics")
FILE_PATH = os.path.join(LOSS_DIR, "loss.jsonl")

with open(FILE_PATH, "r") as f:
    for line in f:
        data = json.loads(line)
        steps.append(data["step"])
        losses.append(data["loss"])

steps = np.array(steps)
losses = np.array(losses)

steps = steps[:CUTOFF]
losses = losses[:CUTOFF]

plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training (cutoff: " + str(CUTOFF) + ")")
plt.savefig(os.path.join(LOSS_DIR, "training_cutoff_" + str(CUTOFF) + ".png"))