import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import MNISTDataloader
from src.nn.unet import UNet
from dataclasses import dataclass

# test Unet on MNIST classification
# yay it works!

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 64
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataloader = DataLoader(MNISTDataloader(train_images_path='/home/willzhao/data/MNIST/train-images.idx3-ubyte', train_labels_path='/home/willzhao/data/MNIST/train-labels.idx1-ubyte', test_images_path='/home/willzhao/data/MNIST/test-images.idx3-ubyte', test_labels_path='/home/willzhao/data/MNIST/test-labels.idx1-ubyte'), batch_size=TrainingConfig.batch_size, shuffle=True)

class MNISTClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.unet = UNet(in_channels, hidden_channels, num_layers)
        self.fc1 = nn.Linear(hidden_channels * 28 * 28, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.unet(x).reshape(x.shape[0], -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

mnist_classifier = MNISTClassifier(in_channels=1, hidden_channels=64, num_layers=2).to(device)
optimizer = optim.Adam(mnist_classifier.parameters(), lr=TrainingConfig.learning_rate)

total_params = sum(p.numel() for p in mnist_classifier.parameters())
print(f'Total number of parameters: {total_params}')


for epoch in range(TrainingConfig.num_epochs):
    correct = 0
    total = 0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        images = images.reshape(-1, 1, 28, 28)
        logits = mnist_classifier(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{TrainingConfig.num_epochs}, Loss: {loss.item()}, Accuracy: {correct / len(train_dataloader.dataset)}")

