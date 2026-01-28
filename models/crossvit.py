"""
CrossViT Model Wrapper
Cross-Attention Multi-Scale Vision Transformer.
"""
import torch.nn as nn
import timm

from models.positional_encodings import get_pe_caller


def create_crossvit(num_classes, encoder_type='sin_cos_1d'):
    """
    Create CrossViT base model with custom positional encoding.
    
    Args:
        num_classes: Number of output classes
        encoder_type: Type of positional encoding
    
    Returns:
        CrossViT model
    """
    model = timm.create_model('crossvit_base_240.in1k', pretrained=True)
    pe_caller = get_pe_caller(encoder_type)
    
    # Replace positional embeddings for both branches
    _, seq_length, hidden_dim = model.pos_embed_0.shape
    model.pos_embed_0 = pe_caller(hidden_dim, seq_length)
    
    _, seq_length, hidden_dim = model.pos_embed_1.shape
    model.pos_embed_1 = pe_caller(hidden_dim, seq_length)
    
    # Replace classification heads
    model.head = nn.ModuleList([
        nn.Linear(in_features=head.in_features, out_features=num_classes, bias=True) 
        for head in model.head
    ])
    
    return model
