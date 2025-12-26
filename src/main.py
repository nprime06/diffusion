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
from data import MNISTDataloader
from nn.resunet import ResUNet
from methods.ddpm.schedule import DDPMScheduler
from training.train_ddpm import train_ddpm
from training.train_fm import train_fm
from run_io import write_run_yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.set_float32_matmul_precision('high')

run_info = {
    "dataset": args.dataset,
    "method": args.method,
    "backbone": args.backbone,
    "device": device
}

@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_steps: int = 100
    batch_size: int = 128
    run_dir: str = args.run_dir
    checkpoint_every: int = 25 # steps

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
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )
# elif args.dataset == 'cifar10':
    # idk
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
        in_channels: int = 1
        hidden_channels: int = 64
        num_layers: int = 2
        embed_dim: int = 64
    
    resunet_config = ResUNetConfig()
    run_info["resunetconfig"] = asdict(resunet_config)
    model = ResUNet(
        in_channels=resunet_config.in_channels, 
        hidden_channels=resunet_config.hidden_channels, 
        num_layers=resunet_config.num_layers, 
        embed_dim=resunet_config.embed_dim,
    ).to(device)
    model = torch.compile(model)
# elif args.backbone == 'vit':
    # idk
else:
    raise ValueError(f"Unsupported backbone: {args.backbone}")

param_count = sum(p.numel() for p in model.parameters())
run_info["param_count"] = int(param_count)

write_run_yaml(args.run_dir, run_info)

if args.method == 'ddpm':
    trained_model = train_ddpm(model, dataloader, scheduler, train_config, device, images_mean, images_std)
elif args.method == 'fm':
    trained_model = train_fm(model, dataloader, fm_config, train_config, device, images_mean, images_std)
else:
    raise ValueError(f"Unsupported method: {args.method}")