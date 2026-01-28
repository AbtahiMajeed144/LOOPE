"""
LOOPE Positional Encodings Module
"""
from .no_pe import get_zero_pos_embed
from .sincos_1d import get_1d_sincos_pos_embed
from .learnable_1d import get_1d_learnable_pos_embed
from .hilbert import get_hilbert_pos_embed, gilbert_2d, generate_hilbert_grid
from .art_pe import ArtPE


def get_pe_caller(encoder_type):
    """
    Get the appropriate positional encoding function based on encoder type.
    
    Args:
        encoder_type: One of 'no_pe', 'sin_cos_1d', 'learnable_1d', 'hilbert'
    
    Returns:
        Function that creates positional embeddings
    """
    pe_callers = {
        'no_pe': get_zero_pos_embed,
        'sin_cos_1d': get_1d_sincos_pos_embed,
        'learnable_1d': get_1d_learnable_pos_embed,
        'hilbert': get_hilbert_pos_embed,
        'pretest': get_1d_sincos_pos_embed,  # Default for testing
    }
    
    if encoder_type not in pe_callers:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose from {list(pe_callers.keys())}")
    
    return pe_callers[encoder_type]
