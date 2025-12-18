import torch.nn as nn
from .convblock import SameConv, DownBlock, UpBlock

class UNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers): # layers > 0 is number of up/down
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        down_blocks_list = [DownBlock(in_channels, hidden_channels)]
        for i in range(num_layers - 1):
            down_blocks_list.append(DownBlock(hidden_channels * 2**i, hidden_channels * 2**(i + 1)))
        self.down_blocks = nn.ModuleList(down_blocks_list)

        self.bot1 = SameConv(hidden_channels * 2**(num_layers - 1), hidden_channels * 2**num_layers)
        self.bot2 = SameConv(hidden_channels * 2**num_layers, hidden_channels * 2**num_layers)

        up_blocks_list = []
        for i in range(num_layers): 
            up_blocks_list.append(UpBlock(hidden_channels * 2**(num_layers - i), hidden_channels * 2**(num_layers - i - 1)))
        self.up_blocks = nn.ModuleList(up_blocks_list)

        # returns (B, hidden, h, w)

    def forward(self, x):
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_connections.append(skip)
        x = self.bot1(x)
        x = self.bot2(x)
        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop())
        return x