import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = d_model // heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, _ = x.shape
        # Linear output is contiguous, so we can safely view -> permute without copies.
        # This avoids `chunk(...).view(...)` pitfalls when the chunk views are non-contiguous.
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)  # each: (B, heads, T, head_dim)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(att)

class CrossAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = d_model // heads

        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, d_model * 2)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x, c):
        B, T, _ = x.shape
        B, t, _ = c.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        kv = self.kv(c).view(B, t, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(dim=0)  # each: (B, heads, t, head_dim)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(att)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_embed, heads, cross_attn=False):
        super().__init__()
        self.ln_self_attn = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttention(d_model, heads)

        self.ln_mlp = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.adaLN = nn.Linear(d_embed, d_model * 6)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

        if cross_attn:
            self.ln_cross_attn = nn.LayerNorm(d_model, elementwise_affine=False)
            self.cross_attn = CrossAttention(d_model, heads)
            self.adaLN_cross = nn.Linear(d_embed, d_model * 3)
            nn.init.zeros_(self.adaLN_cross.weight)
            nn.init.zeros_(self.adaLN_cross.bias)

    def forward(self, res, emb, c=None):
        def modulate(x, scale, shift):
            return x * (1 + scale) + shift

        scale_shift_gate = self.adaLN(emb)
        scale1, shift1, gate1, scale2, shift2, gate2 = scale_shift_gate.chunk(6, dim=-1)

        x = modulate(self.ln_self_attn(res), scale1, shift1)
        x = self.self_attn(x)
        res = res + x * gate1

        if hasattr(self, "cross_attn") and c is not None:
            scale3, shift3, gate3 = self.adaLN_cross(emb).chunk(3, dim=-1)
            x = modulate(self.ln_cross_attn(res), scale3, shift3)
            x = self.cross_attn(x, c)
            res = res + x * gate3

        # MLP residual
        x = modulate(self.ln_mlp(res), scale2, shift2)
        x = self.mlp(x)
        res = res + x * gate2
        return res