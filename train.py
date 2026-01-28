#!/usr/bin/env python
"""
LOOPE Training Script
Main entry point for training Vision Transformers with various positional encodings.

Usage:
    python train.py

Configuration can be modified in config/config.py
"""
import sys
import os

# Add project root to path for local module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
import time
import torch
import torch.optim as optim
import torch.utils.data

# Local imports
from config import config
from data_loaders import load_cifar100_data, load_imagenet100_data
from models import get_model
from training import train_one_epoch, get_validator, param_groups_weight_decay


def main():
    # Suppress warnings if configured
    if config.WARNING == 'supressed':
        warnings.filterwarnings("ignore")
    
    # Generate weight file paths
    BASE_PATH = config.get_weight_path(
        config.MODEL, config.DATASET, config.ENCODER, 
        config.PATCH_SIZE, config.IMG_SIZE
    )
    LAST_PATH = config.get_last_weight_path(
        config.MODEL, config.DATASET, config.ENCODER,
        config.PATCH_SIZE, config.IMG_SIZE
    )
    print(f"Weight file: {BASE_PATH}")
    print(f"{torch.cuda.is_available()}")
    # Device setup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda"
    else:
        device = "cpu"
    print(f"DEVICE WE WILL BE USING IS {device}")
    
    # Load dataset
    if config.DATASET == 'cifar100':
        train_dataset, valid_dataset, test_dataset = load_cifar100_data()
    elif config.DATASET == 'imagenet100':
        train_dataset, valid_dataset, test_dataset = load_imagenet100_data()
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET}")
    
    num_classes = train_dataset.num_cls
    print(f"DATASET LOADED NAMED: {train_dataset.name} WITH NUM OF CLASS: {num_cla sses}")
    
    # Create model
    model = get_model(config.MODEL, num_classes, config.ENCODER, config.IMG_SIZE)
    model = model.to(device)
    
    # DeiT-style LR scaling
    world_size = 1
    global_batch = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS * world_size
    scale = global_batch / 1024.0
    
    max_lr = 5e-4 * scale
    min_lr = 1e-5 * scale
    warmup_lr = 1e-6 * scale
    min_lr = min(min_lr, max_lr)  # Safety: never let min_lr exceed max_lr
    
    # Optimizer: AdamW with DeiT-style weight decay
    optimizer = optim.AdamW(
        param_groups_weight_decay(model, weight_decay=0.05),
        lr=max_lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler: warmup -> cosine (epoch-based)
    warmup_epochs = 5
    total_epochs = config.EPOCHS
    
    start_factor = (warmup_lr / max_lr) if max_lr > 0 else 1.0
    start_factor = max(1e-8, min(1.0, start_factor))
    
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=min_lr
    )
    
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE_VALID,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_VALID,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Get validator function
    validator = get_validator(config.DATASET)
    
    # Training loop
    best_acc = 0.0
    best_epoch = config.START_EPOCH
    train_losses = []
    val_accs = []
    
    for epoch in range(config.START_EPOCH, config.EPOCHS + 1):
        start = time.time()
        
        train_loss = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            grad_accum_steps=config.GRAD_ACCUM_STEPS,
            max_grad_norm=1.0,
            label_smoothing=0.0,
            step_counter=config.STEP_COUNTER,
            dataset_name=config.DATASET
        )
        
        print(f"train loss for epoch {epoch}: {train_loss}")
        train_losses.append(train_loss)
        
        print("validating.....")
        val_acc = validator(model, valid_loader, device)
        val_accs.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            print(f"saving best (epoch={epoch}, val_acc={val_acc:.4f}).....")
            torch.save(model.state_dict(), BASE_PATH)
            
            test_acc = validator(model, test_loader, device)
            print(f"test acc (best checkpoint): {test_acc:.4f}")
        
        torch.save(model.state_dict(), LAST_PATH)
        
        # Scheduler step (once per epoch)
        scheduler.step()
        
        end = time.time()
        print(f"best so far: epoch={best_epoch}, val_acc={best_acc:.4f}")
        print(f"time elapsed (sec): {(end - start):.2f}")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_acc:.4f}% at epoch {best_epoch}")
    print(f"Model saved to: {BASE_PATH}")


if __name__ == "__main__":
    main()
