## File Structure

```text
data/
│
├── MNIST/
├── CIFAR10/
└── etc/

diffusion/
│
├── src/
│   ├── data.py              # data loader
│   ├── embedding.py         # projected sinusoidal proj
│   ├── main.py
│   ├── model.py             # UNet and DiT
│   ├── nn.py                # building blocks
│   ├── test_unet.py
│   ├── test_vit.py
│   │
│   └── training/
│       ├── ddpm.py
│       └── fm.py
│
├── logs/
├── checkpoints/
├── scripts/
├── samples/
│
├── requirements.txt
├── README.md
└── .gitignore
