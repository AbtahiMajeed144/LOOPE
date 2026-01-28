# LOOPE - Learnable Optimal Positional Encoding

**LOOPE** is a research codebase for training Vision Transformers (ViTs) with various custom and experimental positional encoding strategies. The project is designed to be modular, allowing for easy experimentation with different model architectures, datasets, and encoding methods.

## Features

- **Modular Architecture**: Separate modules for models, datasets, training logic, and configurations.
- **Support for Multiple ViT Variants**:
  - ViT-Base (from scratch)
  - DeiT (Data-efficient Image Transformers)
  - BeiT (BERT Pre-Training of Image Transformers)
  - CrossViT (Cross-Attention Multi-Scale ViT)
  - CaiT (Class-Attention in Image Transformers)
- **Experimental Positional Encodings**:
  - Standard 1D Sinusoidal
  - Learnable 1D
  - Hilbert Curve (2D space-filling curve)
  - **Art PE** (Custom learnable content-aware encoding)
  - Fourier PE
  - 2D Sinusoidal (SinCos2D)
  - Conditional PE (CPE)
  - Relative PE (RPE & Cross-RPE)
- **Datasets**: CIFAR-100 and ImageNet-100 support with sophisticated augmentation pipelines (Albumentations).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LOOPE.git
    cd LOOPE
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

```
LOOPE/
├── config/                 # Configuration parameters
│   └── config.py          # Main config file (Model, Dataset, Hyperparams)
├── data_loaders/           # Dataset loading and preprocessing
│   ├── cifar100.py        # CIFAR-100 loader
│   ├── imagenet100.py     # ImageNet-100 loader
│   └── transforms.py      # Custom augmentations
├── models/                 # Model definitions
│   ├── positional_encodings/  # PE implementations (Hilbert, Art PE, etc.)
│   ├── vit_base.py        # ViT implementation
│   └── ... (other model wrappers)
├── training/               # Training loop and utilities
│   ├── trainer.py         # Training engine
│   └── validator.py       # Validation logic
├── experiments/            # Visualization and analysis scripts
├── train.py                # Main entry point for training
└── requirements.txt        # Python dependencies
```

## Usage

### 1. Configure the Training

Modify `config/config.py` to set your desired experiment parameters:

```python
# Select Model and Encoding
MODEL = 'vit_b_16'          # Options: 'vit_b_16', 'vit_art', 'Fourier', 'SinCos2D', ...
DATASET = 'imagenet100'     # Options: 'cifar100', 'imagenet100'
ENCODER = 'sin_cos_1d'      # Options: 'sin_cos_1d', 'hilbert', 'no_pe', ...

# Hyperparameters
BATCH_SIZE = 96             # Global batch size
EPOCHS = 500
LEARNING_RATE = ...         # Configured via MAX_LR/MIN_LR with DeiT scaling
```

### 2. Prepare Datasets

The code expects datasets to be in specific locations or will download/process them.
- **CIFAR-100**: Handled automatically by `load_cifar100_data` (uses `sklearn` to split if needed).
- **ImageNet-100**: Expects a folder structure (e.g., `Datasets/ImageNeT-100`) or will attempt to load cached splits from `Datasets/imagenet100`.

### 3. Run Training

Execute the main script:

```bash
python train.py
```

The script will:
1.  Load the configuration.
2.  Initialize the dataset and model.
3.  Train for the specified number of epochs.
4.  Save model checkpoints to the project root (e.g., `weights_...pth`).
5.  Log training loss and validation accuracy.

## Available Models

- `vit_b_16`: Standard ViT-Base
- `vit_art`: ViT with Art PE
- `Fourier`: ViT with Fourier PE
- `SinCos2D`: ViT with 2D Sin/Cos PE
- `CPEViT`: Conditional Positional Encoding ViT
- `RPE`: Relative Positional Encoding ViT
- `deit_b_16`, `beit_b_16`, `crossvit_base_240`, `cait_s24_224`

## License

[MIT License](LICENSE)
