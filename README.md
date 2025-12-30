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
```

---

## Notes for scaling this repo (DiT + SD VAE + DDIM + rectified flow + more)

You said you’re moving to **bigger models**, switching to **DiT (Transformer) backbones**, importing a **Stable Diffusion VAE**, and adding **more advanced training/sampling** (DDIM, rectified flow, etc.). These notes are the recommended way to evolve this repo so new methods/backbones/samplers slot in cleanly.

## Current structure (today)

```text
diffusion/
├── src/
│   ├── main.py                 # CLI entrypoint (currently also the "registry/config")
│   ├── data.py                 # MNIST + CIFAR10 datasets (CPU normalized)
│   ├── embedding.py            # TimeEmbedding
│   ├── run_io.py               # run.yaml + metrics + checkpoints + GIF writer
│   ├── nn/                     # backbones (UNet/ResUNet; dit.py is currently empty placeholder)
│   ├── methods/                # method math (DDPM schedule/loss/sampler, FM loss/sampler)
│   └── training/               # training loops per method (train_ddpm.py, train_fm.py)
├── scripts/                    # cluster submit scripts
└── logs/                       # run artifacts
```

## Recommended target architecture (to support DiT + SD VAE + many samplers/methods)

Right now, “a method” is implicitly spread across:
- `methods/<name>/*` (loss + sampler + schedule)
- `training/train_<name>.py` (optimizer loop + logging + sampling)
- `main.py` (wiring/config)

That works for 2 methods, but it becomes brittle when you add: DDIM, EDM/Heun samplers, rectified flow, latent diffusion (VAE), different conditionings, etc.

The scaling move is: **separate “what you train” (method) from “how you train” (trainer), and separate “how you sample” (sampler) from the method math**.

### Target module layout

```text
src/
├── cli/
│   └── main.py                # parse args, load config, call train/eval
├── config/
│   ├── defaults.py            # dataclasses (or pydantic) with sane defaults
│   └── presets/               # yaml presets per experiment (mnist_ddpm_resunet.yaml, ...)
├── data/
│   ├── datasets.py            # MNIST/CIFAR10 datasets (raw images)
│   └── transforms.py          # normalize / resize / augment; VAE preprocessing
├── models/
│   ├── unet/                  # ResUNet etc.
│   ├── dit/                   # DiT backbone (+ patchify, AdaLN, etc.)
│   └── conditioning.py        # time/class/text conditioning utilities
├── vae/
│   ├── interface.py           # encode/decode API
│   └── sd_vae.py              # Stable Diffusion VAE wrapper (optional dep on diffusers)
├── methods/
│   ├── ddpm/                  # forward process + training objective (epsilon, v, x0)
│   ├── rectified_flow/        # RF training objective (velocity field)
│   └── registry.py            # method factory (string -> class)
├── samplers/
│   ├── ddim.py                # DDIM sampler (for DDPM-style models)
│   ├── ddpm.py                # ancestral sampler
│   ├── ode.py                 # generic ODE solver wrapper (euler/heun/rk4)
│   └── registry.py            # sampler factory
├── training/
│   ├── trainer.py             # ONE training loop (optimizer, EMA, grad accum, log hooks)
│   └── ema.py                 # EMA weights (recommended for bigger models)
└── infra/
    ├── run_io.py              # run.yaml, metrics, checkpoints, samples
    └── device.py              # autocast/compile knobs, grad checkpointing flags
```

You don’t need to do this refactor all at once—see “Suggested refactor steps” below.

## Key interfaces (the extension points)

If you add just these boundaries, you can implement DDIM/RF/etc without rewriting training:

### 1) `Backbone` / `Denoiser` interface

Unify all backbones (ResUNet, DiT) behind one forward signature:
- **input**: `x` (image or latent), `t` (float in [0,1] or int timestep), `cond` (optional dict)
- **output**: prediction tensor shaped like `x` (eps/v/x0 depending on objective)

Today `ResUNet.forward(x, t, c=None)` already fits this shape; for DiT you’ll likely switch to something like:
- `cond = {"class": labels, "text": text_emb, ...}`
- classifier-free guidance = “drop cond” by setting fields to `None` or using a learned null token.

### 2) `Method` = training objective (not trainer)

Make a small object per method that defines:
- **how to sample a training timestep** \(t\)
- **how to generate the training input** \(x_t\)
- **what the model is supposed to predict** (epsilon, v, x0, velocity)
- **how to compute loss**

This keeps DDPM vs RF differences out of the main training loop.

### 3) `Sampler` = inference integrator

Sampling shouldn’t live inside each method’s training loop.

Instead:
- a `Sampler` consumes `(model, method, noise/init_state, cond, num_steps, cfg_scale, ...)`
- and returns either final samples or a history tensor (for GIFs)

Then DDIM becomes “just another sampler” for the DDPM method, and RK/Heun becomes “just another sampler” for RF/ODE-style methods.

### 4) `Autoencoder` (VAE) wrapper

Treat the VAE as an optional preprocessing/postprocessing module:
- training data: image \(x\) -> latent \(z\)
- sampling: latent \(z\) -> image \(x\)

Keep this outside the method math so DDPM/RF can run in image space or latent space with minimal changes.

## Notes for importing a Stable Diffusion VAE

Practical integration tips:
- **Dependency**: easiest path is HuggingFace `diffusers` `AutoencoderKL` (optional dependency; keep it out of your core requirements if you want the repo lightweight).
- **Latent shape**: SD VAE maps image \(3×H×W\) to latent \(4×(H/8)×(W/8)\). \(H,W\) must typically be divisible by 8.
- **Latent scaling**: Stable Diffusion commonly uses a scaling factor of ~`0.18215`. In practice:
  - encode: `z = vae.encode(x).latent_dist.sample() * 0.18215`
  - decode: `x = vae.decode(z / 0.18215).sample`
  (Exact API depends on implementation; keep the scale as a constant in your wrapper.)
- **Dataset resolution mismatch**: MNIST/CIFAR10 are tiny. You can:
  - train in pixel space until your DiT + samplers are correct, then switch to latent diffusion later, or
  - upsample to a VAE-friendly resolution (multiples of 8; often 256/512) knowing this changes the task.

## Notes for adding DDIM / advanced samplers

With your existing DDPM scheduler, DDIM is a natural next sampler:
- It uses the same model prediction (usually epsilon) and the same \(\alpha_t\) schedule,
- but uses a deterministic update (or partially stochastic with `eta`).

Implementation recommendation:
- keep schedule logic in a `Scheduler` (alphas, alpha_cumprod, etc.)
- implement `samplers/ddim.py` using only scheduler getters + model prediction
- don’t put DDIM inside `methods/ddpm/sampler.py` if you want to mix and match samplers.

## Notes for rectified flow (RF) / flow matching

Your current `methods/fm/*` is already close to a rectified-flow style objective (linear interpolation between noise and data + velocity target).

To evolve it cleanly:
- rename/clarify what the model predicts (`v` field) and standardize time to float in [0,1]
- move Euler (and future Heun/RK) into `samplers/ode.py` so RF can share integrators
- keep “how to build \(x_t\) and target \(v\)” in the RF method module

## Suggested refactor steps (incremental, low-risk)

If you want the smallest set of changes that unlocks everything:

1) **Unify training loops**
   - create `training/trainer.py` with one loop
   - move “method-specific loss call” behind a `Method.loss(...)` hook
   - keep your existing logging (`run_io.py`) as-is

2) **Pull sampling out of training**
   - training only saves checkpoints + optionally calls a sampler hook
   - samplers live in `samplers/*` and can be swapped by config

3) **Add a lightweight registry**
   - `methods.registry.get_method(name, cfg)`
   - `models.registry.get_model(name, cfg)`
   - `samplers.registry.get_sampler(name, cfg)`
   This keeps `main.py` from becoming a giant if/else tree.

4) **Add VAE wrapper**
   - implement `vae/interface.py` + `vae/sd_vae.py`
   - initially run VAE encode/decode only in sampling to verify plumbing
   - then switch training data to latents (optionally precompute latents to disk for speed)

5) **Implement DiT**
   - add patchify/unpatchify and a simple DiT block
   - match the same forward signature as ResUNet so methods/samplers don’t care

## Logging / artifacts conventions (keep doing this)

Your `logs/<method>/<method>_<backbone>_<timestamp>/` layout is solid. Keep:
- `run.yaml`: full config snapshot (method/model/data/sampler/training)
- `checkpoints/step_XXXXXXXX.pt`: model + optimizer (+ EMA if you add it)
- `metrics/loss.jsonl`: append-only training metrics
- `samples/step_XXXXXXXX.gif`: qualitative sampling snapshots

If you move to latent diffusion, consider also logging:
- decoded images (for human viewing)
- a small grid of raw latents (for debugging only)
