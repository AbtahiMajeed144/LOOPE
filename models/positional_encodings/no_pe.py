"""
Zero/No Positional Encoding
Returns a fixed zero tensor - effectively no positional information.
"""
import torch
import torch.nn as nn


def get_zero_pos_embed(embed_dim, seq_length):
    """
    Create zero positional embeddings (no positional encoding).
    
    Args:
        embed_dim: Embedding dimension
        seq_length: Sequence length (including CLS token)
    
    Returns:
        nn.Parameter with requires_grad=False
    """
    return nn.Parameter(
        torch.zeros((1, seq_length, embed_dim), dtype=torch.float32), 
        requires_grad=False
    )
