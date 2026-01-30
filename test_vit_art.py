
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from models.vit_art import ViT_Art

def test_vit_art():
    print("Initializing ViT_Art...")
    try:
        model = ViT_Art(num_classes=10, image_size=224)
        print("Initialization successful.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("Running forward pass...")
    try:
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")
        if out.shape == (2, 10):
            print("Test PASSED: Output shape is correct.")
        else:
            print(f"Test FAILED: Expected (2, 10), got {out.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_vit_art()
