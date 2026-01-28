"""
ViT with 2D Sinusoidal Positional Encoding
Vision Transformer using 2D sin/cos positional encoding.
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import ViTForImageClassification

from models.positional_encodings import get_zero_pos_embed


class Sin2DPositionalEncoding(nn.Module):
    """2D Sinusoidal positional encoding using x,y coordinates."""
    
    def __init__(self, num_patches, dim):
        super().__init__()
        self.num_patches = num_patches
        self.dim = dim
        
        # Frequency scaling term
        self.register_buffer('div_term', torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)))
        
        grid_size = int(np.sqrt(self.num_patches))  # Typically 14x14 for 196 patches
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        positions = np.stack([xx.flatten(), yy.flatten()], axis=-1)  # (num_patches, 2)
        
        self.register_buffer('normalized_positions', torch.tensor(positions, dtype=torch.float32))
        self.register_buffer('cls_embed', torch.zeros((1, self.dim)))
    
    def forward(self, x):
        positions = self.normalized_positions.unsqueeze(-1)  # (196, 2, 1)
        pe = torch.zeros(self.num_patches, self.dim, device=x.device)
        pe[:, 0::2] = torch.sin(positions[:, 0] * self.div_term[: self.dim // 2]) + torch.sin(positions[:, 1] * self.div_term[: self.dim // 2])
        pe[:, 1::2] = torch.cos(positions[:, 0] * self.div_term[: self.dim // 2]) + torch.cos(positions[:, 1] * self.div_term[: self.dim // 2])
        pe = torch.cat([self.cls_embed, pe], dim=0).unsqueeze(0)
        return pe + x


class ViTSinCos2D(nn.Module):
    """ViT model with 2D sinusoidal positional encoding."""
    
    def __init__(self, num_classes):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        _, self.seq_length, self.hidden_dim = self.model.vit.embeddings.position_embeddings.shape
        
        # Replace with zero position embeddings
        self.model.vit.embeddings.position_embeddings = get_zero_pos_embed(self.hidden_dim, self.seq_length)
        
        num_patches = self.model.vit.embeddings.patch_embeddings.num_patches
        dim = self.model.config.hidden_size
        
        # Create 2D sinusoidal positional encoding
        self.pos_embedding = Sin2DPositionalEncoding(num_patches=num_patches, dim=dim)
        self.patch = self.model.vit.embeddings.patch_embeddings
        self.encoder = self.model.vit.encoder
        self.layernorm = self.model.vit.layernorm
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=num_classes, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
    
    def forward(self, x):
        n = x.shape[0]
        patched = self.patch(x)
        batched_cls_token = self.cls_token.expand(n, -1, -1)
        patched = torch.cat([batched_cls_token, patched], dim=1)
        patched = self.pos_embedding(patched)
        patched = self.encoder(patched)
        patched = self.layernorm(patched.last_hidden_state)
        patched = patched[:, 0]
        patched = self.classifier(patched)
        return patched
