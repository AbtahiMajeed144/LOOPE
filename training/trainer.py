"""
Trainer Module
Training function with gradient accumulation support.
"""
import gc
import torch
import torch.nn as nn

from .utils import AverageMeter


def train_one_epoch(train_loader, model, optimizer, epoch, device, 
                    grad_accum_steps=1, max_grad_norm=1.0, label_smoothing=0.0,
                    step_counter=50, dataset_name='imagenet100'):
    """
    Train model for one epoch.
    
    Args:
        train_loader: Training data loader
        model: Model to train
        optimizer: Optimizer
        epoch: Current epoch number
        device: Device to train on
        grad_accum_steps: Number of gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        label_smoothing: Label smoothing factor
        step_counter: Print every N steps
        dataset_name: Name of dataset for loss selection
    
    Returns:
        Average training loss for the epoch
    """
    model.train()

    grad_accum_steps = max(1, int(grad_accum_steps))
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch [{epoch}] - Learning Rate: {current_lr:.6f}")

    losses = AverageMeter()
    bce_crit = nn.BCEWithLogitsLoss()
    ce_crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing > 0 else nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    micro_steps = 0
    total_batches = len(train_loader)

    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(images)
        logits = out.logits if hasattr(out, "logits") else out

        if dataset_name == "new_dataset":
            loss = bce_crit(logits, labels.float())
        else:
            targets = labels.argmax(dim=1).long() if labels.ndim > 1 else labels.long()
            loss = ce_crit(logits, targets)

        losses.update(loss.detach().item(), labels.size(0))

        accum_count += 1
        micro_steps += 1

        # Scale by actual number of micro-batches in this update
        if (idx == total_batches - 1) and (accum_count != grad_accum_steps):
            scale = accum_count
        else:
            scale = grad_accum_steps

        (loss / scale).backward()

        do_step = (accum_count == grad_accum_steps) or (idx == total_batches - 1)
        if do_step:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        if micro_steps % step_counter == 0:
            print(f"micro-step: {micro_steps} | avg loss: {losses.avg:.6f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return losses.avg
