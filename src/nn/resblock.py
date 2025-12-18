import torch
import torch.nn as nn

class ResBlock(nn.Module): # expects time conditioning, need to change adaGN for non time conditioning
    def __init__(self, fan_in, fan_out, embed_dim, groups=16): 
        super().__init__()
        self.gn1 = nn.GroupNorm(min(groups, fan_in), fan_in, affine=True)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(fan_in, fan_out, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(groups, fan_out), fan_out, affine=False) # adaGN time embed on second GN
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(fan_out, fan_out, kernel_size=3, padding=1)
        self.act3 = nn.SiLU()

        if fan_in != fan_out:
            self.skip_conv = nn.Conv2d(fan_in, fan_out, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

        self.time_proj = nn.Linear(embed_dim, fan_out * 2)

    def forward(self, x, t_emb): # t_emb is (B, embed_dim)
        res = self.skip_conv(x)
        x = self.gn2(self.conv1(self.act1(self.gn1(x))))
        scale, shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.conv2(self.act2(x))
        x = x + res
        x = self.act3(x)
        return x

class DownResBlock(nn.Module):
    def __init__(self, fan_in, fan_out, embed_dim):
        super().__init__()
        self.res = ResBlock(fan_in, fan_out, embed_dim)
        self.down = nn.Conv2d(fan_out, fan_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip

class UpResBlock(nn.Module):
    def __init__(self, fan_in, fan_out, embed_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(fan_in, fan_out, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock(fan_out * 2, fan_out, embed_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return x