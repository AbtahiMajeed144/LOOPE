"""
Art PE (LOOPE Custom Positional Encoding)
Learnable optimal positional encoding based on image content.
"""
import torch
import torch.nn as nn
import numpy as np

from .hilbert import gilbert_2d


class ArtPE(nn.Module):
    """
    Adaptive Positional Encoding that learns position offsets based on image content
    combined with Hilbert curve base positions.
    """
    
    def __init__(self, seq_length, hidden_dim, img_size=224, patch_size=16):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Coordinate grid
        self.register_buffer(
            'cord', 
            torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, img_size, dtype=torch.float),
                torch.linspace(-1, 1, img_size, dtype=torch.float),
                indexing='ij'
            ))
        )
        
        # Convolutional network to learn position adjustments from image+coordinates
        self.art_conv = nn.Sequential(
            nn.Conv2d(5, 32, 16, 16, 0), nn.GELU(),
            nn.Conv2d(32, 16, 5, 1, 2), nn.GELU(),
            nn.Conv2d(16, 8, 5, 1, 2), nn.GELU(),
            nn.Conv2d(8, 4, 5, 1, 2), nn.GELU(),
            nn.Conv2d(4, 1, 5, 1, 2), nn.GELU(),
            nn.BatchNorm2d(1), nn.Flatten(),
            nn.Linear((img_size // patch_size) ** 2, self.seq_length - 1, bias=False), 
            nn.Sigmoid()
        )
        
        # Base position from Hilbert curve
        self.register_buffer(
            'wrap_pos_hilbert',
            torch.Tensor(gilbert_2d(
                (img_size // patch_size), 
                (img_size // patch_size)
            )).reshape(self.seq_length - 1).unsqueeze(1)
        )
        
        # Division term for sinusoidal encoding
        self.register_buffer(
            'div_term',
            torch.exp(
                torch.arange(0, hidden_dim, 2, dtype=torch.float32).unsqueeze(0) * 
                -(np.log(10000.0) / hidden_dim)
            )
        )
        
        # CLS token positional embedding
        self.cls_token = nn.Parameter(torch.zeros((1, self.hidden_dim)))

    def forward(self, x):
        n = x.shape[0]
        
        # Concatenate image with coordinate grid
        out = self.art_conv(
            torch.cat([x, self.cord.unsqueeze(0).expand(n, -1, -1, -1)], dim=1)
        ).unsqueeze(2)
        
        # Scale to [-1, 1] and add Hilbert base position
        out = 2 * out - 1
        out = out + self.wrap_pos_hilbert
        
        # Apply sinusoidal encoding
        sin_part = torch.sin(out * self.div_term)
        cos_part = torch.cos(out * self.div_term)
        
        pos_embedding = torch.empty(
            (n, self.seq_length, self.hidden_dim), 
            dtype=self.wrap_pos_hilbert.dtype, 
            device=x.device
        )
        pos_embedding[:, 1:, 0::2] = sin_part
        pos_embedding[:, 1:, 1::2] = cos_part
        pos_embedding[:, :1, :] = self.cls_token.unsqueeze(0).expand(n, -1, -1)
        
        return pos_embedding
