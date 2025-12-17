## File Structure

```text
data/ (on cloud)
├── MNIST/
├── CIFAR10/
└── etc/

diffusion/
├── src/
│   ├── data.py
│   ├── embedding.py
│   │
│   ├── nn/
│   │   ├── convblock.py
│   │   ├── resblock.py
│   │   ├── unet.py
│   │   ├── resunet.py
│   │   ├── ?.py
│   │   └── dit.py
│   │
│   ├── methods/
│   │   ├── ddpm/
│   │   │   ├── loss.py
│   │   │   ├── schedule.py
│   │   │   └── sampler.py
│   │   └── fm/
│   │       ├── loss.py
│   │       └── sampler.py
│   │
│   ├── training/
│   │   ├── train_ddpm.py
│   │   └── train_fm.py
│   │
│   ├── main.py
│   │
│   └── test_arch/
│       ├── test_unet.py
│       └── test_vit.py
│
├── logs/
├── checkpoints/
├── scripts/
├── samples/
│
├── requirements.txt
├── README.md
└── .gitignore

conceptual flow: 
train -> save stuff in logs, checkpoints
samples -> model from checkpoints