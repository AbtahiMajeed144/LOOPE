"""
Visualization and Analysis Experiments
Positional encoding visualization and similarity analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from numpy.linalg import norm


def radial_smooth(array, center=None, bin_width=1.0, return_std=False):
    """
    Apply radial smoothing to an array based on distance from center.
    
    Args:
        array: 2D numpy array to smooth
        center: Center point (a, b). If None, uses array center.
        bin_width: Width of radial bins
        return_std: If True, also return standard deviation per bin
    
    Returns:
        smoothed: Smoothed array
        values: (optional) Per-bin statistics
    """
    nrows, ncols = array.shape
    if center is None:
        center = (nrows / 2, ncols / 2)
    a, b = center
    y, x = np.indices((nrows, ncols))
    r = np.sqrt((y - a)**2 + (x - b)**2)
    bin_indices = np.floor(r / bin_width).astype(int)
    
    if return_std:
        values = np.zeros(np.unique(bin_indices).shape[0])
    
    smoothed = np.zeros_like(array)
    for bin_val in np.unique(bin_indices):
        mask = (bin_indices == bin_val)
        if return_std:
            values[bin_val] = array[mask].std()
        smoothed[mask] = array[mask].mean()
    
    if return_std:
        return smoothed, values
    return smoothed


def running_mean(x, N):
    """Compute running mean of array x with window size N."""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def compute_similarity_map(pos_embedding, center=(0, 0)):
    """
    Compute cosine similarity between center position and all other positions.
    
    Args:
        pos_embedding: Positional embedding array of shape (H, W, D)
        center: Center position tuple (y, x)
    
    Returns:
        similarity: 2D similarity map
    """
    cen = center
    similarity = ((pos_embedding * pos_embedding[cen]).sum(-1)) / (norm(pos_embedding, axis=-1) * norm(pos_embedding[cen]))
    return similarity


def visualize_positional_encoding(model, image, device, save_path=None):
    """
    Visualize positional encoding similarity map.
    
    Args:
        model: Model with pos_embedding attribute
        image: Input image tensor
        device: Device to use
        save_path: Optional path to save figure
    """
    img = image.unsqueeze(0).to(device)
    pos = model.pos_embedding(img).squeeze().detach().cpu().numpy()[1:].reshape(14, 14, 768)
    
    cen = (0, 0)
    similarity = compute_similarity_map(pos, cen)
    
    plt.imshow(similarity, cmap='hot')
    plt.title('Positional Encoding Similarity')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_with_smoothing(similarity, center=(0, 0), upscale=True):
    """
    Visualize similarity with radial smoothing.
    
    Args:
        similarity: 2D similarity array
        center: Center point for radial smoothing
        upscale: If True, upscale to 224x224
    """
    if upscale:
        array_224 = zoom(similarity, (224 / 14, 224 / 14), order=1)
        center = (center[0] * 16 + 8, center[1] * 16 + 8)
        smoothed, values = radial_smooth(array_224, center=center, bin_width=1.0, return_std=True)
    else:
        smoothed, values = radial_smooth(similarity, center=center, bin_width=1.0, return_std=True)
    
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(similarity if not upscale else array_224, cmap='hot')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(smoothed, cmap='hot')
    plt.title('Radially Smoothed')
    plt.show()
    
    return values


def plot_training_curves(train_losses, val_accs):
    """
    Plot training loss and validation accuracy curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_accs: List of validation accuracies per epoch
    """
    x = np.arange(0, len(val_accs), 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, train_losses, label="train_loss", color="blue", linestyle="-")
    plt.plot(x, val_accs, label="val_accuracy", color="red", linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_image_with_patches(image_tensor, grid_size=14, img_size=224):
    """
    Visualize image with patch grid overlay.
    
    Args:
        image_tensor: Image tensor of shape (C, H, W)
        grid_size: Number of patches per dimension
        img_size: Image size
    """
    img = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    cell_size = img_size // grid_size
    
    for i in range(grid_size + 1):
        ax.axhline(i * cell_size, color='white', linewidth=0.5)
        ax.axvline(i * cell_size, color='white', linewidth=0.5)
    
    ax.axis('off')
    plt.show()


def visualize_art_pe_components(model, image, device):
    """
    Visualize Art PE (LOOPE) internal components.
    
    Args:
        model: Model with Art PE
        image: Input image tensor
        device: Device to use
    """
    np.set_printoptions(linewidth=np.inf, precision=2, suppress=True)
    
    x = image.unsqueeze(0).to(device)
    cord = model.pos_embedding.cord
    art_conv = model.pos_embedding.art_conv
    
    out = art_conv(torch.cat([x, cord.unsqueeze(0).expand(1, -1, -1, -1)], dim=1)).squeeze().detach().cpu().numpy()
    out = 2 * out - 1
    
    wrap_pos_hilbert = model.pos_embedding.wrap_pos_hilbert.squeeze().detach().cpu().numpy()
    
    print("Learned offset (scaled):")
    print(out.reshape(14, 14))
    
    print("\nHilbert base position:")
    print(wrap_pos_hilbert.reshape(14, 14))
    
    print("\nCombined position:")
    print((out + wrap_pos_hilbert).reshape(14, 14))
