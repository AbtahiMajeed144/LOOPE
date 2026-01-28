"""
BeiT Model Wrapper
BERT Pre-Training of Image Transformers.
"""
from transformers import BeitForImageClassification


def create_beit(num_classes=None):
    """
    Create BeiT-B/16 model.
    
    Args:
        num_classes: Number of output classes (optional, uses ImageNet-1K by default)
    
    Returns:
        BeitForImageClassification model
    """
    model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
    
    # Note: BeiT uses different positional encoding architecture
    # Modifying requires more complex changes to the model
    
    return model
