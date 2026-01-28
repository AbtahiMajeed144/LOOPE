"""
Validator Module
Validation functions for different dataset types.
"""
import torch
import torch.nn as nn
import numpy as np


def validate_new_dataset(model, valid_loader, device):
    """
    Validation for new_dataset (multi-label classification).
    
    Args:
        model: Model to validate
        valid_loader: Validation data loader
        device: Device to use
    
    Returns:
        Average accuracy across all metrics
    """
    model.eval()
    total = 0
    dist_comp = 0
    orientation = 0
    area_comp = 0
    vec_sum = 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            outputs = model(images)
            try:
                outputs = nn.Sigmoid()(outputs.logits).detach().cpu().numpy()
            except:
                outputs = nn.Sigmoid()(outputs).detach().cpu().numpy()
            outputs = np.where(outputs > 0.5, 1, 0)
            labels = np.where(labels > 0.5, 1, 0)
            dist_comp += (outputs[:, :2] == labels[:, :2]).sum()
            orientation += (outputs[:, 2] == labels[:, 2]).sum()
            area_comp += (outputs[:, 3:5] == labels[:, 3:5]).sum()
            vec_sum += (outputs[:, 5] == labels[:, 5]).sum()
            total += labels.shape[0]
    
    dist_comp_acc = 100 * dist_comp / (total * 2)
    orientation_acc = 100 * orientation / total
    area_comp_acc = 100 * area_comp / (2 * total)
    vec_sum_acc = 100 * vec_sum / total
    avg_accuracy = (dist_comp_acc + orientation_acc + area_comp_acc + vec_sum_acc) / 4
    
    print(f'Distance comparison accuracy: {dist_comp_acc:.4f}%')
    print(f'Orientation accuracy: {orientation_acc:.4f}%')
    print(f'Area comparison accuracy: {area_comp_acc:.4f}%')
    print(f'Vector sum accuracy: {vec_sum_acc:.4f}%')
    print(f'Average Accuracy: {avg_accuracy:.4f}%')
    
    return avg_accuracy


def validate_regular(model, valid_loader, device):
    """
    Standard validation for classification datasets.
    
    Args:
        model: Model to validate
        valid_loader: Validation data loader
        device: Device to use
    
    Returns:
        Validation accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(images)
            logits = out if torch.is_tensor(out) else out.logits  # [B, C]

            preds = logits.argmax(dim=1)  # [B]

            # Support both one-hot labels [B, C] and index labels [B]
            if labels.ndim > 1:
                targets = labels.argmax(dim=1)
            else:
                targets = labels.long()

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / max(1, total)
    print(f'Validation/Test Accuracy: {accuracy:.4f}%')
    return accuracy


def get_validator(dataset_name):
    """
    Get appropriate validator function based on dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Validator function
    """
    if dataset_name == 'new_dataset':
        return validate_new_dataset
    else:
        return validate_regular
