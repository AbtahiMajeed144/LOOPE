"""
CaiT Model Wrapper
Class-Attention in Image Transformers.
"""
import torch.nn as nn
import timm

from models.positional_encodings import get_pe_caller


def create_cait(num_classes, encoder_type='sin_cos_1d'):
    """
    Create CaiT-S24 model with custom positional encoding.
    
    Args:
        num_classes: Number of output classes
        encoder_type: Type of positional encoding
    
    Returns:
        CaiT model
    """
    model = timm.create_model('cait_s24_224.fb_dist_in1k', pretrained=True)
    pe_caller = get_pe_caller(encoder_type)

    # Get model's embedding dimensions
    _, seq_length, hidden_dim = model.pos_embed.shape

    # Replace positional embeddings
    temp = pe_caller(hidden_dim, seq_length + 1)
    model.pos_embed = nn.Parameter(temp[:, 1:, :], requires_grad=temp.requires_grad)
    model.cls_token = nn.Parameter(temp[:, :1, :], requires_grad=temp.requires_grad)

    # Modify classifier layer
    model.head = nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=True)
    
    return model
