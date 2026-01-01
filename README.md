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
│   ├── run_io.py
│   ├── main.py
│   │
│   └── test_arch/
│       ├── test_unet.py
│       └── test_vit.py
│
├── scripts/
│   ├── submit_main.sh
│   └── main_job.sh
│
├── logs/
│   └── method/
│       └── method_backbone_date_time/
│           ├── checkpoints/
│           ├── metrics/
│           ├── samples/
│           ├── run.yaml
│           ├── .out
│           └── .err
│
├── requirements.txt
├── README.md
└── .gitignore

how to:
* set environment vars
* call submit_main.sh

example run:
* create /logs/ddpm/ddpm_unet_2025-12-17_23-59-59
* save in run.yaml hyperparams (model arch, optimizer, etc.) and other config (e.g. schedule) for reproducibility
* checkpoints: save state dict every x steps/epochs
* metrics: save .jsonl of loss per step
* samples: save samples every y steps/epochs