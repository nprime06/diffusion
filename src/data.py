import numpy as np
import idx2numpy as idx
import os
import math
import torch
from torch.utils.data import Dataset

def load_mnist_images_labels(image_path, label_path, ):
    images = torch.from_numpy(idx.convert_from_file(open(image_path, 'rb')).copy()).float() # copy to avoid modifying the original data
    labels = torch.from_numpy(idx.convert_from_file(open(label_path, 'rb')).copy()).long()
    return images, labels

class MNISTDataloader(Dataset): # everything is on cpu
    def __init__(self, test_images_path, test_labels_path, train_images_path, train_labels_path):
        self.test_images, self.test_labels = load_mnist_images_labels(test_images_path, test_labels_path)
        self.train_images, self.train_labels = load_mnist_images_labels(train_images_path, train_labels_path)
        self.images = torch.cat((self.test_images, self.train_images)) # (N, 28, 28)
        self.labels = torch.cat((self.test_labels, self.train_labels))

        self.images_mean = self.images.mean(dim=0, keepdim=True).float()
        self.images_std = self.images.std(dim=0, keepdim=True).float()
        self.images = (self.images - self.images_mean) / (self.images_std + 1e-6).float()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx] # (28, 28), (1,)
    
    def get_mean_std(self):
        return self.images_mean, self.images_std # (1, 28, 28), (1, 28, 28)


class CIFAR10Dataloader(Dataset): # everything is on cpu
    def __init__(self, root):
        # Lazy import: if torchvision is mis-installed it can segfault at import time,
        # so we avoid importing it unless CIFAR-10 is actually used.
        from torchvision.datasets import CIFAR10

        train_ds = CIFAR10(root=root, train=True, download=True)
        train_images = torch.from_numpy(train_ds.data.copy()).permute(0, 3, 1, 2).float() / 255.0 # (N, 3, 32, 32)
        train_labels = torch.tensor(train_ds.targets, dtype=torch.long) # (N,)

        test_ds = CIFAR10(root=root, train=False, download=True)
        test_images = torch.from_numpy(test_ds.data.copy()).permute(0, 3, 1, 2).float() / 255.0 # (N, 3, 32, 32)
        test_labels = torch.tensor(test_ds.targets, dtype=torch.long) # (N,)

        self.images = torch.cat((train_images, test_images), dim=0) # (N, 3, 32, 32)
        self.labels = torch.cat((train_labels, test_labels), dim=0) # (N,)

        self.images_mean = self.images.mean(dim=0, keepdim=True).float() # (1, 3, 32, 32)
        self.images_std = self.images.std(dim=0, keepdim=True).float()   # (1, 3, 32, 32)
        self.images = (self.images - self.images_mean) / (self.images_std + 1e-6).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx] # (3, 32, 32), (1,)

    def get_mean_std(self):
        return self.images_mean, self.images_std # (1, 3, 32, 32), (1, 3, 32, 32)


class AFHQDataloader(Dataset):
    def __init__(self, root, image_size=512, augment=True, color_jitter=False):
        # Lazy import: if torchvision is mis-installed it can segfault at import time,
        # so we avoid importing it unless AFHQ is actually used.
        from torchvision.datasets import ImageFolder
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        from torch.utils.data import ConcatDataset

        # images: (3, image_size, image_size) in float [0, 1]
        # note afhq is big
        self.root = root
        self.image_size = int(image_size)
        self.augment = augment
        self.color_jitter = color_jitter

        tfms = [transforms.Lambda(lambda img: img.convert("RGB"))]
        if augment:
            tfms.append(transforms.RandomHorizontalFlip(p=0.5))
            if color_jitter:
                tfms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02))
            tfms.append(
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.9, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=InterpolationMode.BICUBIC,
                )
            )
        else:
            tfms.append(transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC))
            tfms.append(transforms.CenterCrop(self.image_size))

        tfms.append(transforms.ToTensor())
        transform = transforms.Compose(tfms)

        # AFHQ is small and we're training generative models, so we want as much data
        # as possible: combine any of train/val/test if they exist.
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        test_dir = os.path.join(root, "test")

        if os.path.isdir(train_dir) or os.path.isdir(val_dir) or os.path.isdir(test_dir):
            dss = []
            if os.path.isdir(train_dir):
                dss.append(ImageFolder(train_dir, transform=transform))
            if os.path.isdir(val_dir):
                dss.append(ImageFolder(val_dir, transform=transform))
            if len(dss) == 0:
                raise FileNotFoundError(f"No AFHQ split directories found under: {root}")
            self.ds = ConcatDataset(dss)
        else:
            # If `root` already points directly to an ImageFolder-style directory
            # (root/{class_name}/*), just use that.
            self.ds = ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]  # image: (3, image_size, image_size) float in [0,1]
        return image, torch.tensor(label, dtype=torch.long)

    def get_mean_std(self):
        return (torch.zeros(1, 3, self.image_size, self.image_size, dtype=torch.float32), torch.ones(1, 3, self.image_size, self.image_size, dtype=torch.float32))

# compress dataloader
# save logvar
# clamp var
# make sure to use sd_latent_scale

SD_LATENT_SCALE = 0.18215

class encodedDataloader(Dataset):
    def __init__(self, pre_encoded_dataloader, vae, latent_shape, device, clamp_logvar=(-30.0, 20.0)):
        self.latent_shape = tuple(latent_shape)
        total_samples = len(pre_encoded_dataloader.dataset)

        self.mu = torch.empty((total_samples, *self.latent_shape), dtype=torch.float32)
        self.logvar = torch.empty_like(self.mu)
        self.labels = torch.empty((total_samples,), dtype=torch.long)

        log_scale = 2.0 * math.log(SD_LATENT_SCALE)

        sample_offset = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pre_encoded_dataloader):
                images = images.to(device, non_blocking=True)
                images = (images * 2.0 - 1.0).clamp(-1.0, 1.0)  # VAE expects inputs in [-1, 1]

                latent_dist = vae.encode(images).latent_dist
                mu = latent_dist.mean * SD_LATENT_SCALE
                logvar = latent_dist.logvar + log_scale
                logvar = logvar.clamp(min=clamp_logvar[0], max=clamp_logvar[1])

                batch_size = mu.shape[0]
                batch_slice = slice(sample_offset, sample_offset + batch_size)

                self.mu[batch_slice].copy_(mu.detach().cpu())
                self.logvar[batch_slice].copy_(logvar.detach().cpu())
                self.labels[batch_slice].copy_(labels.detach().cpu().long())

                sample_offset += batch_size

    def __len__(self):
        return self.mu.shape[0]

    def __getitem__(self, idx):
        mean = self.mu[idx]
        logvar = self.logvar[idx]
        latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return latents, self.labels[idx]