"""
DeiT Model Wrapper
Data-efficient Image Transformer with custom positional encoding.
"""
import torch.nn as nn
from transformers import ViTForImageClassification

from models.positional_encodings import get_pe_caller


def create_deit(num_classes, encoder_type='sin_cos_1d'):
    """
    Create DeiT-B/16 model with custom positional encoding.
    
    Args:
        num_classes: Number of output classes
        encoder_type: Type of positional encoding
    
    Returns:
        ViTForImageClassification model with DeiT weights
    """
    model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
    _, seq_length, hidden_dim = model.vit.embeddings.position_embeddings.shape
    
    pe_caller = get_pe_caller(encoder_type)
    model.vit.embeddings.position_embeddings = pe_caller(hidden_dim, seq_length)
    model.classifier = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    
    return model
