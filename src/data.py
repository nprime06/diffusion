import numpy as np
import idx2numpy as idx
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