"""
Conditional Positional Encoding ViT (CPE-ViT)
ViT with depth-wise convolution based conditional positional encoding.
"""
import torch
import torch.nn as nn
from transformers import ViTModel


class ConditionalPositionalEncoding(nn.Module):
    """Conditional positional encoding using depth-wise convolution."""
    
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)

    def forward(self, x):
        B, N, D = x.shape  # (Batch, Num Tokens, Embedding Dim)
        H = W = int((N - 1) ** 0.5)  # Ignore CLS token for reshaping

        cls_token = x[:, 0:1, :]  # Extract CLS token
        x = x[:, 1:, :]
        
        # Reshape for convolution
        cpe = x.transpose(1, 2).reshape(B, D, H, W)  # (B, D, H, W)
        cpe = self.conv(cpe).flatten(2).transpose(1, 2)  # Apply CPE

        x = x + cpe  # Add CPE to the features

        return torch.cat([cls_token, x], dim=1)  # Reinsert CLS token


class CPEViT(nn.Module):
    """ViT with Conditional Positional Encoding."""
    
    def __init__(self, num_classes, model_name="google/vit-base-patch16-224"):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(model_name)
        self.vit.embeddings.position_embeddings = None  # Remove default positional embeddings

        embed_dim = self.vit.config.hidden_size
        num_patches = (self.vit.config.image_size // self.vit.config.patch_size) ** 2

        self.encoder1 = self.vit.encoder.layer[0]
        self.cpe = ConditionalPositionalEncoding(embed_dim, num_patches)
        self.remaining_encoders = nn.ModuleList(self.vit.encoder.layer[1:])
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.vit.embeddings.patch_embeddings(x)  # (B, 196, D)

        batch_size = x.shape[0]
        cls_tokens = self.vit.embeddings.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, D)

        x = self.encoder1(x)[0]  # Extract tensor from tuple

        x = self.cpe(x)  # Apply Conditional Positional Encoding

        for layer in self.remaining_encoders:
            x = layer(x)[0]  # Extract tensor from tuple

        cls_token_final = x[:, 0]  # (B, D)
        out = self.fc(cls_token_final)  # (B, num_classes)
        
        return out
