"""
ViT with Fourier Positional Encoding
Vision Transformer using learned Fourier features for positional encoding.
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import ViTForImageClassification

from models.positional_encodings import get_zero_pos_embed


class FourierPositionalEncoding(nn.Module):
    """Fourier-based positional encoding with learnable frequencies."""
    
    def __init__(self, num_patches, dim, fourier_dim=768, hidden_dim=32, groups=1):
        super().__init__()
        self.num_patches = num_patches  # N = 196
        self.fourier_dim = fourier_dim  # |F| = 768
        self.hidden_dim = hidden_dim    # |H| = 32
        self.dim = dim                  # D = 768
        self.groups = groups            # G = 1
        self.M = 2                      # M = 2D positional values

        grid_size = int(np.sqrt(self.num_patches))  # Typically 14x14 for 196 patches
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        positions = np.stack([xx.flatten(), yy.flatten()], axis=-1)  # (num_patches, 2)
        
        # Reshape to (N, G, M) = (196, 1, 2)
        self.register_buffer('positions', torch.tensor(positions, dtype=torch.float32).unsqueeze(1))
        
        # Learnable Fourier weights Wr ∈ R^(|F|/2, M), sampled from N(0, γ^-2)
        self.Wr = nn.Parameter(torch.randn(fourier_dim//2, self.M))  # (384, 2)
        
        # MLP layers
        self.fc1 = nn.Linear(fourier_dim, hidden_dim, bias=True)  # (768, 32)
        self.fc2 = nn.Linear(hidden_dim, dim // groups, bias=True)  # (32, 256)
        self.activation = nn.GELU()
        self.cls_embed = nn.Parameter(torch.zeros((1, self.dim)))

    def forward(self, x):
        """Compute Fourier-based positional encoding."""
        # Compute Fourier features F = [cos(XWr^T); sin(XWr^T)]
        proj = torch.matmul(self.positions, self.Wr.T)  # (196, 1, 384)
        
        F = torch.empty((self.num_patches, self.groups, self.fourier_dim), dtype=proj.dtype, device=x.device)
        F[:, :, 0::2] = torch.sin(proj)
        F[:, :, 1::2] = torch.cos(proj)
        
        # Pass through MLP: Y = GeLU(FW1 + B1)W2 + B2
        Y = self.fc2(self.activation(self.fc1(F)))  # (196, 1, 256)
        
        # Reshape Y to (N, D) = (196, 768)
        PEX = Y.reshape(self.num_patches, self.dim)
        pos = torch.cat([self.cls_embed, PEX], dim=0)
        
        return pos + x


class ViTFourier(nn.Module):
    """ViT model with Fourier positional encoding."""
    
    def __init__(self, num_classes):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        _, self.seq_length, self.hidden_dim = self.model.vit.embeddings.position_embeddings.shape
        
        # Replace with zero position embeddings
        self.model.vit.embeddings.position_embeddings = get_zero_pos_embed(self.hidden_dim, self.seq_length)
        
        num_patches = self.model.vit.embeddings.patch_embeddings.num_patches
        dim = self.model.config.hidden_size
        
        # Create Fourier positional encoding
        self.pos_embedding = FourierPositionalEncoding(num_patches=num_patches, dim=dim, fourier_dim=768, hidden_dim=32, groups=1)
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
