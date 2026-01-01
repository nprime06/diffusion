import torch
import torch.nn as nn
from embedding import TimeEmbedding
from nn.transformer import TransformerBlock

class DiT(nn.Module):
    def __init__(self, d_model, d_embed, heads, num_layers, image_shape, p=8, num_classes=0, cross_attn=False): # p patch size
        super().__init__()
        if len(image_shape) != 3:
            raise ValueError(f"image_shape must be (C,H,W), got {image_shape}")
        if p <= 0:
            raise ValueError(f"p must be > 0, got {p}")
        in_channels, H, W = image_shape
        if H % p != 0 or W % p != 0:
            raise ValueError(f"image H,W must be divisible by p. Got H={H}, W={W}, p={p}")

        self.image_shape = tuple(image_shape)
        self.p = int(p)
        self.grid_h = H // p
        self.grid_w = W // p
        self.num_patches = self.grid_h * self.grid_w
        self.in_channels = in_channels

        self.time_embedding = TimeEmbedding(embed_dim=d_embed)
        self.class_embedding = nn.Embedding(num_classes+1, d_embed) if num_classes > 0 else None
        self.patch_in = nn.Conv2d(in_channels, d_model, kernel_size=p, stride=p, padding=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_embed, heads, cross_attn=cross_attn) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(d_model)

        # project tokens back to patches, then unpatchify
        self.patch_out = nn.Linear(d_model, in_channels * p * p)
        nn.init.zeros_(self.patch_out.weight)
        nn.init.zeros_(self.patch_out.bias)

    def forward(self, x, t, c=None, cross_attn_tokens=None): 
        """
        x: (B, C, H, W)
        t: (B,)
        c: (B,) optional class labels (0 is reserved for unconditional)
        cross_attn_tokens: (B, Tc, D) optional conditioning tokens if cross-attn blocks are enabled
        """
        B, C, H, W = x.shape
        if (C, H, W) != self.image_shape:
            raise ValueError(f"Expected x shape (B,{self.image_shape[0]},{self.image_shape[1]},{self.image_shape[2]}), got {x.shape}")

        # patches -> tokens
        x = self.patch_in(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, T, D)
        x = x + self.pos_embedding

        # conditioning
        emb = self.time_embedding(t)  # (B, d_embed)
        if self.class_embedding is not None:
            if c is None:
                c = torch.zeros_like(t, dtype=torch.long, device=x.device)
            emb = emb + self.class_embedding(c)

        # transformer
        for block in self.blocks:
            x = block(x, emb, c=cross_attn_tokens)
        x = self.ln_final(x)

        # tokens -> patches -> image
        x = self.patch_out(x)  # (B, T, C*p*p)
        x = x.view(B, self.grid_h, self.grid_w, self.in_channels, self.p, self.p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, self.in_channels, H, W)
        return x
