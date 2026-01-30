"""
ImageNet-100 Dataset Module
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.config import IMG_SIZE, XIMG_SIZE, TVT_SPLIT, IMAGENET_MEAN, IMAGENET_STD


def _imagenet_norm_keep_div255(img, **kwargs):
    """
    ImageNet normalization compatible with later /255.0 division.
    output = (x - mean*255)/std  and later /255 => (x/255 - mean)/std  (correct)
    """
    img = img.astype(np.float32)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32) * 255.0
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (img - mean) / std


class ImageNet100Dataset(Dataset):
    """ImageNet-100 dataset class with optional RAM preloading."""
    
    def __init__(self, files, cls_dict, mode='train', preload_ram=True):
        self.mode = mode
        self.name = 'imagenet_100'
        self.cls_dict = cls_dict
        self.num_cls = len(cls_dict)
        
        self.labels = [int(file[1]) for file in files]
        self.preload_ram = preload_ram
        
        if self.preload_ram:
            # Preload into a single contiguous uint8 array to minimize overhead
            N = len(files)
            self.images = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            for i, file in enumerate(files):
                p = os.path.normpath(file[2])
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Failed to read image: {p}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # uint8
                
                # Safety: if any image isn't exactly 224x224, resize once during preload
                if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                
                self.images[i] = img
        else:
            self.images = [file[2] for file in files]  # paths
        
        # Transforms
        if self.mode == 'train':
            self.transforms = A.Compose([
                A.RandomResizedCrop(
                    size=(IMG_SIZE, IMG_SIZE),
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.3333333),
                    interpolation=cv2.INTER_CUBIC,
                    p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                ], p=0.8),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.2),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=int(IMG_SIZE * 0.25),
                    max_width=int(IMG_SIZE * 0.25),
                    min_holes=1,
                    min_height=int(IMG_SIZE * 0.10),
                    min_width=int(IMG_SIZE * 0.10),
                    fill_value=0,
                    p=0.25
                ),
                A.Lambda(image=_imagenet_norm_keep_div255),
                ToTensorV2(),
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(XIMG_SIZE, XIMG_SIZE, interpolation=cv2.INTER_CUBIC),
                A.CenterCrop(IMG_SIZE, IMG_SIZE),
                A.Lambda(image=_imagenet_norm_keep_div255),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.preload_ram:
            img = self.images[idx]  # uint8 RGB
        else:
            path = os.path.normpath(self.images[idx])
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = self.transforms(image=img)['image'] / 255.0
        
        label = self.labels[idx]
        temp = torch.zeros(self.num_cls).float()
        temp[label] = 1.0
        return img, temp


def _load_splits(dirpath):
    """Load cached train/valid/test splits from directory."""
    tr = np.load(os.path.join(dirpath, "train_files.npz"), allow_pickle=True)["data"]
    va = np.load(os.path.join(dirpath, "valid_files.npz"), allow_pickle=True)["data"]
    te = np.load(os.path.join(dirpath, "test_files.npz"), allow_pickle=True)["data"]
    return tr, va, te


def load_imagenet100_data(root_dir="data_loaders/ImageNeT-100",
                          split_dir="data_loaders/imagenet100"):
    """
    Load ImageNet-100 dataset with train/valid/test splits.
    
    Args:
        root_dir: Root directory containing class folders
        split_dir: Directory to save/load cached splits
    
    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    os.makedirs(split_dir, exist_ok=True)
    
    # Try load cached splits
    loaded = False
    for cand in [
        "/kaggle/input/imagenet100-splits/imagenet100",  # optional (if you created/uploaded)
        split_dir
    ]:
        try:
            if os.path.isfile(os.path.join(cand, "train_files.npz")):
                train_files, valid_files, test_files = _load_splits(cand)
                print(f"Successfully preloaded train/valid/test splits from: {cand}")
                loaded = True
                break
        except:
            pass
    
    if not loaded:
        # Discover class folders (e.g., n01440764)
        classes_synset = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        ])
        cls2idx = {c: i for i, c in enumerate(classes_synset)}
        
        # Build file tuples: (placeholder, label_index, image_path)
        files = []
        exts = ("*.JPEG", "*.JPG", "*.jpeg", "*.jpg", "*.png")
        for c in classes_synset:
            cdir = os.path.join(root_dir, c)
            for ext in exts:
                for p in glob.glob(os.path.join(cdir, ext)):
                    files.append((None, cls2idx[c], p))
        
        # Stratified TVT split
        labels = [f[1] for f in files]
        vt_ratio = (TVT_SPLIT[1] + TVT_SPLIT[2]) / 100.0
        
        train_files, vt_files = train_test_split(
            files,
            test_size=vt_ratio,
            random_state=42,
            stratify=labels
        )
        
        vt_labels = [f[1] for f in vt_files]
        test_ratio_inside_vt = TVT_SPLIT[2] / (TVT_SPLIT[1] + TVT_SPLIT[2])
        
        valid_files, test_files = train_test_split(
            vt_files,
            test_size=test_ratio_inside_vt,
            random_state=42,
            stratify=vt_labels
        )
        
        # Cache splits
        np.savez(os.path.join(split_dir, "train_files.npz"), data=np.array(train_files, dtype=object))
        np.savez(os.path.join(split_dir, "valid_files.npz"), data=np.array(valid_files, dtype=object))
        np.savez(os.path.join(split_dir, "test_files.npz"), data=np.array(test_files, dtype=object))
        
        print(f"Created and cached splits in: {split_dir}")
        print("Classes discovered:", len(classes_synset))
    
    # Infer number of classes from labels
    all_labels = np.array([f[1] for f in train_files], dtype=np.int64)
    num_classes = int(all_labels.max()) + 1
    print("total classes:", num_classes)
    
    classes = [i for i in range(num_classes)]
    train_dataset = ImageNet100Dataset(train_files, classes, 'train')
    valid_dataset = ImageNet100Dataset(valid_files, classes, 'valid')
    test_dataset = ImageNet100Dataset(test_files, classes, 'test')
    
    return train_dataset, valid_dataset, test_dataset
