import torch
import torch.optim as optim
from dataclasses import dataclass
from methods.ddpm.loss import loss


def train_ddpm(model, dataloader, scheduler, train_config): # scheduler, dataloader on cpu; model on device
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)

    step = 0
    for epoch in range(train_config.num_epochs):
        for images, labels in dataloader:
            images, labels = images.to(model.device), labels.to(model.device)
            loss_val = loss(model, images, scheduler)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Step {step}, Loss: {loss_val.item()}")
            step += 1
            
    return model