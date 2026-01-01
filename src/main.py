import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--backbone', type=str, required=True)
args = parser.parse_args()

from dataclasses import asdict, dataclass
import torch
from torch.utils.data import DataLoader
from data import MNISTDataloader, CIFAR10Dataloader, AFHQDataloader
from nn.resunet import ResUNet
from nn.dit import DiT
from nn.sdvae import get_vae
from methods.ddpm.schedule import DDPMScheduler
from training.train_ddpm import train_ddpm
from training.train_fm import train_fm
from run_io import write_run_yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

run_info = {
    "dataset": args.dataset,
    "method": args.method,
    "backbone": args.backbone,
    "device": device
}

@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    max_steps: int = 1000
    batch_size: int = 128
    cfg_proportion: float = 0.8
    run_dir: str = args.run_dir
    early_checkpoint_every: int = 500 # steps
    num_early_checkpoints: int = 3
    late_checkpoint_every: int = 25000 # steps

train_config = TrainConfig()
run_info["trainconfig"] = asdict(train_config)

if args.dataset == 'mnist':
    dataset = MNISTDataloader(
        train_images_path='/home/willzhao/data/MNIST/train-images.idx3-ubyte',
        train_labels_path='/home/willzhao/data/MNIST/train-labels.idx1-ubyte',
        test_images_path='/home/willzhao/data/MNIST/test-images.idx3-ubyte',
        test_labels_path='/home/willzhao/data/MNIST/test-labels.idx1-ubyte',
    )
    images_mean, images_std = dataset.get_mean_std()
    image_shape = (1, 28, 28)
    in_channels = 1 # pixel channels
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    num_classes = 10
    vae = None
elif args.dataset == 'cifar10':
    dataset = CIFAR10Dataloader(root="/home/willzhao/data/CIFAR_10/")
    images_mean, images_std = dataset.get_mean_std()
    image_shape = (3, 32, 32)
    in_channels = 3 # pixel channels
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    num_classes = 10
    vae = None
elif args.dataset == 'afhq':
    dataset = AFHQDataloader(root="/home/willzhao/data/afhq")
    images_mean, images_std = dataset.get_mean_std()
    # Dataloader returns RGB images; we VAE-encode inside the training loop.
    image_shape = (3, 512, 512)
    in_channels = 4 # latent channels
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    num_classes = 3
    vae = get_vae(device=device)
    run_info["vae"] = "stabilityai/sd-vae-ft-mse"
    run_info["latent_shape"] = (4, 64, 64)
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

if args.method == 'ddpm':
    @dataclass
    class DDPMConfig:
        beta_start: float = 1e-4
        beta_end: float = 0.02
        num_steps: int = 1000
    
    ddpm_config = DDPMConfig()
    run_info["ddpmconfig"] = asdict(ddpm_config)
    scheduler = DDPMScheduler(
        beta_start=ddpm_config.beta_start, 
        beta_end=ddpm_config.beta_end, 
        num_steps=ddpm_config.num_steps,
    )
elif args.method == 'fm': 
    @dataclass
    class FMConfig:
        num_steps: int = 100
        sampler: str = 'euler'

    fm_config = FMConfig()
    run_info["fmconfig"] = asdict(fm_config)
else:
    raise ValueError(f"Unsupported method: {args.method}")

if args.backbone == 'unet':
    @dataclass
    class ResUNetConfig:
        # If None, we infer this from the selected dataset's `image_shape`.
        # (MNIST: 1, CIFAR-10: 3)
        in_channels: int
        hidden_channels: int = 256
        num_layers: int = 3
        embed_dim: int = 256
    resunet_config = ResUNetConfig(in_channels=in_channels)
    run_info["resunetconfig"] = asdict(resunet_config)
    model = ResUNet(
        in_channels=resunet_config.in_channels, 
        hidden_channels=resunet_config.hidden_channels, 
        num_layers=resunet_config.num_layers, 
        embed_dim=resunet_config.embed_dim,
        num_classes=num_classes,
    ).to(device)
    model = torch.compile(model)
elif args.backbone == 'dit':
    # d_model, d_embed, heads, num_layers, image_shape, p, num_classes=0, cross_attn=False)
    @dataclass
    class DiTConfig:
        image_shape: tuple[int, int, int]
        d_model: int = 256
        d_embed: int = 128
        heads: int = 8
        num_layers: int = 6
        p: int = 4
        cross_attn: bool = False
    dit_config = DiTConfig(image_shape=image_shape)
    run_info["ditconfig"] = asdict(dit_config)
    model = DiT(
        d_model=dit_config.d_model,
        d_embed=dit_config.d_embed,
        heads=dit_config.heads,
        num_layers=dit_config.num_layers,
        image_shape=dit_config.image_shape,
        p=dit_config.p,
        num_classes=num_classes,
        cross_attn=dit_config.cross_attn,
    ).to(device)
    model = torch.compile(model)
else:
    raise ValueError(f"Unsupported backbone: {args.backbone}")

param_count = sum(p.numel() for p in model.parameters())
run_info["param_count"] = int(param_count)

write_run_yaml(args.run_dir, run_info)

if args.method == 'ddpm':
    trained_model = train_ddpm(model, dataloader, scheduler, train_config, device, image_shape, images_mean, images_std, vae)
elif args.method == 'fm':
    trained_model = train_fm(model, dataloader, fm_config, train_config, device, image_shape, images_mean, images_std, vae)
else:
    raise ValueError(f"Unsupported method: {args.method}")