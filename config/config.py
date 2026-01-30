"""
LOOPE Training Configuration
Session parameters and hyperparameters for model training.
"""

# Model and Dataset Selection
MODEL = 'vit_b_16'          # Options: 'vit_b_16', 'deit_b_16', 'beit_b_16', 'crossvit_base_240', 'vit_art', 'Fourier', 'SinCos2D', 'CPEViT', 'RPE', 'cait_s24_224', 'Cross-RPE'
DATASET = 'imagenet100'     # Options: 'cifar100', 'imagenet100', 'new_dataset'
ENCODER = 'sin_cos_1d'      # Options: 'sin_cos_1d', 'learnable_1d', 'hilbert', 'no_pe'

# Image and Patch Settings
PATCH_SIZE = 16             # 16, 32, 8
IMG_SIZE = 224
XIMG_SIZE = 256

# Data Split
TVT_SPLIT = (80, 10, 10)    # Train, Validation, Test percentages

# Training Hyperparameters
BATCH_SIZE = 128                 # Used for train/valid/test loaders and LR scaling
BATCH_SIZE_TRAIN = BATCH_SIZE   # Alias for compatibility
BATCH_SIZE_VALID = BATCH_SIZE   # Alias for compatibility
EPOCHS = 500
START_EPOCH = 1

# Learning Rate (will be scaled by DeiT-style LR scaling)
MAX_LR = 0.0001
MIN_LR = 0.0001 / 20

# Optimizer
OPTIM = 'adam'

# Gradient Accumulation
GRAD_ACCUM_STEPS = 8

# Device
DEVICE = 'cuda'  # 'gpu' or 'cpu'

# Logging
STEP_COUNTER = 50           # Print every N steps
WARNING = 'supressed'       # 'supressed' or 'enabled'

# Initialize best tracking
best_acc = 0.0
best_epoch = 0

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_weight_path(model, dataset, encoder, patch_size, img_size):
    """Generate weight file path based on configuration."""
    return f"weights_(model-{model})_(data-{dataset})_(enc-{encoder})_(p-{patch_size})_(im-{img_size}).pth"


def get_last_weight_path(model, dataset, encoder, patch_size, img_size):
    """Generate last checkpoint weight file path."""
    return f"weights(last)_(model-{model})_(data-{dataset})_(enc-{encoder})_(p-{patch_size})_(im-{img_size}).pth"
