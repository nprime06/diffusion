import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_size=64, max_freq = 2, min_freq = 0): # freq is log scale
        super().__init__()
        self.powers_base = 2 * torch.pi * torch.logspace(min_freq, max_freq, steps=embed_size // 2, base=10).unsqueeze(0) # (1, embed_size // 2)
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        self.min_freq = min_freq

        self.proj = nn.Linear(embed_size, embed_dim)
        self.act = nn.silu()

    def forward(self, t):
        powers_base_cast = self.powers_base * t
        sin_embed = torch.sin(powers_base_cast)
        cos_embed = torch.cos(powers_base_cast)
        embedding = torch.cat((sin_embed, cos_embed), dim=1) # (b, embed_size)
        return self.act(self.proj(embedding))
