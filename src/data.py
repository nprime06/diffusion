import numpy as np
import idx2numpy as idx
import torch
from torch.utils.data import Dataset, DataLoader

def load_mnist_images_labels(image_path, label_path, ):
    images = torch.from_numpy(idx.convert_from_file(open(image_path, 'rb')).copy()).float() # copy to avoid modifying the original data
    labels = torch.from_numpy(idx.convert_from_file(open(label_path, 'rb')).copy()).long() # copy to avoid modifying the original data

    images_mean = images.mean(dim=0, keepdim=True)
    images_std = images.std(dim=0, keepdim=True)
    images = (images - images_mean) / (images_std + 1e-6)

    return images, labels

class MNISTDataloader(Dataset): # everything is on cpu
    def __init__(self, test_images_path, test_labels_path, train_images_path, train_labels_path):
        self.test_images, self.test_labels = load_mnist_images_labels(test_images_path, test_labels_path)
        self.train_images, self.train_labels = load_mnist_images_labels(train_images_path, train_labels_path)
        self.images = np.concatenate((self.test_images, self.train_images)) # (N, 28, 28)
        self.labels = np.concatenate((self.test_labels, self.train_labels))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx] # (28, 28), (1,)