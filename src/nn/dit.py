import torch.nn as nn
import torch.nn.functional as F
from nn.transformer import TransformerBlock

class DiT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, embed_dim, num_classes=0):