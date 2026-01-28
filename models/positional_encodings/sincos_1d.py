"""
1D Sinusoidal Positional Encoding
Standard transformer positional encoding using sin/cos functions.
"""
import torch
import torch.nn as nn
import numpy as np


def get_1d_sincos_pos_embed(embed_dim, seq_length):
    """
    Create 1D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        seq_length: Sequence length (including CLS token)
    
    Returns:
        nn.Parameter with requires_grad=False
    """
    position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(1)  # (seq_length, 1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(np.log(10000.0) / embed_dim))
    
    pos_embed = torch.zeros((seq_length, embed_dim), dtype=torch.float32)
    pos_embed[:, 0::2] = torch.sin(position * div_term)  # apply sin to even indices
    pos_embed[:, 1::2] = torch.cos(position * div_term)  # apply cos to odd indices
    
    return nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)  # (1, seq_length, embed_dim)
