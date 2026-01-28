"""
1D Learnable Positional Encoding
Trainable positional embeddings initialized with normal distribution.
"""
import torch
import torch.nn as nn


def get_1d_learnable_pos_embed(embed_dim, seq_length):
    """
    Create 1D learnable positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        seq_length: Sequence length (including CLS token)
    
    Returns:
        nn.Parameter with requires_grad=True
    """
    return nn.Parameter(
        torch.empty(1, seq_length, embed_dim).normal_(std=0.02), 
        requires_grad=True
    )
