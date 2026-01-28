"""
LOOPE Models Module
Vision Transformer variants with different positional encodings.
"""
from .vit_base import create_vit_base
from .vit_art import ViTArt
from .vit_fourier import ViTFourier, FourierPositionalEncoding
from .vit_sincos2d import ViTSinCos2D, Sin2DPositionalEncoding
from .cpe_vit import CPEViT, ConditionalPositionalEncoding
from .rpe_vit import create_rpe_vit, GlobalRelativePositionBias, RifatAttention
from .deit import create_deit
from .beit import create_beit
from .crossvit import create_crossvit
from .cait import create_cait
from .cross_rpe import create_cross_rpe_vit, CrossMethodRPE

from .positional_encodings import (
    get_pe_caller,
    get_zero_pos_embed,
    get_1d_sincos_pos_embed,
    get_1d_learnable_pos_embed,
    get_hilbert_pos_embed,
    ArtPE,
)


def get_model(model_name, num_classes, encoder_type='sin_cos_1d', img_size=224):
    """
    Factory function to create models by name.
    
    Args:
        model_name: One of 'vit_b_16', 'vit_art', 'Fourier', 'SinCos2D', 'CPEViT', 
                    'RPE', 'deit_b_16', 'beit_b_16', 'crossvit_base_240', 'cait_s24_224', 'Cross-RPE'
        num_classes: Number of output classes
        encoder_type: Type of positional encoding for applicable models
        img_size: Input image size
    
    Returns:
        Model instance
    """
    if model_name == 'vit_b_16':
        return create_vit_base(num_classes, img_size, encoder_type)
    elif model_name == 'vit_art':
        return ViTArt(num_classes)
    elif model_name == 'Fourier':
        return ViTFourier(num_classes)
    elif model_name == 'SinCos2D':
        return ViTSinCos2D(num_classes)
    elif model_name == 'CPEViT':
        return CPEViT(num_classes)
    elif model_name == 'RPE':
        return create_rpe_vit(num_classes, encoder_type)
    elif model_name == 'deit_b_16':
        return create_deit(num_classes, encoder_type)
    elif model_name == 'beit_b_16':
        return create_beit(num_classes)
    elif model_name == 'crossvit_base_240':
        return create_crossvit(num_classes, encoder_type)
    elif model_name == 'cait_s24_224':
        return create_cait(num_classes, encoder_type)
    elif model_name == 'Cross-RPE':
        return create_cross_rpe_vit()
    else:
        raise ValueError(f"Unknown model: {model_name}")
