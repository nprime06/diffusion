import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--backbone', type=str, required=True)
args = parser.parse_args()

print("args: ", args)

from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from data import MNISTDataloader
from nn.unet import UNet
from methods.ddpm.schedule import DDPMScheduler
from training.train_ddpm import train_ddpm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 64
    run_dir: str = args.run_dir


if args.dataset == 'mnist':
    dataloader = DataLoader(MNISTDataloader(train_images_path='/home/willzhao/data/MNIST/train-images.idx3-ubyte', train_labels_path='/home/willzhao/data/MNIST/train-labels.idx1-ubyte', test_images_path='/home/willzhao/data/MNIST/test-images.idx3-ubyte', test_labels_path='/home/willzhao/data/MNIST/test-labels.idx1-ubyte'), batch_size=TrainConfig.batch_size, shuffle=True)
# elif args.dataset == 'cifar10':
    # dataloader = DataLoader(CIFAR10Dataloader(train_images_path='/home/willzhao/data/CIFAR10/train-images.idx3-ubyte', train_labels_path='/home/willzhao/data/CIFAR10/train-labels.idx1-ubyte', test_images_path='/home/willzhao/data/CIFAR10/test-images.idx3-ubyte', test_labels_path='/home/willzhao/data/CIFAR10/test-labels.idx1-ubyte'), batch_size=TrainConfig.batch_size, shuffle=True)

if args.method == 'ddpm':
    scheduler = DDPMScheduler(beta_start=1e-4, beta_end=0.02, num_steps=1000)
# elif args.method == 'fm':
    # idk

if args.backbone == 'unet':
    model = UNet(in_channels=1, hidden_channels=64, num_layers=2).to(device)
# elif args.backbone == 'vit':
    # model = ViT(in_channels=1, hidden_channels=64, num_layers=2).to(device)

print(f"Param count: {sum(p.numel() for p in model.parameters())}")

trained_model = train_ddpm(model, dataloader, scheduler, TrainConfig)



