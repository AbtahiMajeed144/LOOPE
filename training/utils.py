"""
Training Utilities
Helper classes and functions for training.
"""
import torch


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def param_groups_weight_decay(model, weight_decay=0.05):
    """
    Create parameter groups with DeiT-style weight decay.
    No weight decay on biases/norm and cls/pos tokens.
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay value
    
    Returns:
        List of parameter groups for optimizer
    """
    decay, no_decay = [], []
    skip_wd_names = ("position_embeddings", "pos_embed", "cls_token", "dist_token")
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim == 1
            or name.endswith(".bias")
            or ("norm" in name.lower())
            or any(k in name for k in skip_wd_names)
        ):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
