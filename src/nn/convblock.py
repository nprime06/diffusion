import torch
import torch.nn as nn

class SameConv(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.conv = nn.Conv2d(fan_in, fan_out, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))

class DownBlock(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.res1 = SameConv(fan_in, fan_out)
        self.res2 = SameConv(fan_out, fan_out)
        self.down = nn.Conv2d(fan_out, fan_out, kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.res2(self.res1(x))
        skip = x
        x = self.act(self.down(x))
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(fan_in, fan_out, kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU()
        self.res1 = SameConv(fan_out * 2, fan_out)  # because of skip concat
        self.res2 = SameConv(fan_out, fan_out)

    def forward(self, x, skip):
        x = self.act(torch.cat([self.up(x), skip], dim=1))
        x = self.res2(self.res1(x))
        return x