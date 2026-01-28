"""
ViT Base Model (from scratch)
Vision Transformer trained from scratch with custom positional encoding.
"""
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

from models.positional_encodings import get_pe_caller


def create_vit_base(num_classes, img_size=224, encoder_type='sin_cos_1d'):
    """
    Create a ViT-B/16 model from scratch with custom positional encoding.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        encoder_type: Type of positional encoding
    
    Returns:
        ViTForImageClassification model
    """
    config = ViTConfig(
        image_size=img_size,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=num_classes,
        # DeiT-style: usually keep dropout ~0 and use drop-path + strong aug + mixup/cutmix
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    
    # Stochastic depth (drop-path)
    if hasattr(config, "drop_path_rate"):
        config.drop_path_rate = 0.1
    
    # Scratch initialization
    model = ViTForImageClassification(config)
    
    # Replace position embeddings
    _, seq_length, hidden_dims = model.vit.embeddings.position_embeddings.shape
    pe_caller = get_pe_caller(encoder_type)
    new_pe = pe_caller(hidden_dims, seq_length)
    
    if isinstance(new_pe, torch.Tensor) and not isinstance(new_pe, nn.Parameter):
        new_pe = nn.Parameter(new_pe)
    
    model.vit.embeddings.position_embeddings = new_pe
    
    return model
