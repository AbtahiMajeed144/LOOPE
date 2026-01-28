"""
CIFAR-100 Dataset Module
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config.config import IMG_SIZE, XIMG_SIZE, TVT_SPLIT


class Cifar100Dataset(Dataset):
    """CIFAR-100 dataset class with albumentations transforms."""
    
    def __init__(self, files, cls_dict, mode='train'):
        self.mode = mode
        self.name = 'cifar_100'
        self.cls_dict = cls_dict
        self.num_cls = len(cls_dict)
        
        print(files[33][2].shape)
        self.images = [np.array(file[2]).reshape((3, 32, 32)).transpose((1, 2, 0)) for file in files]
        self.labels = [file[1] for file in files]
        
        if self.mode == 'train':
            self.transforms = A.Compose([
                A.Resize(XIMG_SIZE, XIMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomCrop(IMG_SIZE, IMG_SIZE),
                ToTensorV2()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = self.transforms(image=image)['image'] / 255.00
        temp = torch.zeros(self.num_cls).float()
        temp[label] = 1.0
        
        return image, temp


def unpickle(file):
    """Unpickle CIFAR-100 data file."""
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_cifar100_data(split_dir="/kaggle/input/cifar100-splits/cifar100/",
                       train_dir='/kaggle/input/cifar100/train'):
    """
    Load CIFAR-100 dataset with train/valid/test splits.
    
    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    try:
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Directory '{split_dir}' does not exist!")
        train_files = np.load(split_dir + "train_files.npz", allow_pickle=True)["data"]
        valid_files = np.load(split_dir + "valid_files.npz", allow_pickle=True)["data"]
        test_files = np.load(split_dir + "test_files.npz", allow_pickle=True)["data"]
        print('Successfully preloaded train, valid, test splits.....')
    except:
        train_dict = unpickle(train_dir)
        lst = list(train_dict.keys())
        length = len(train_dict[lst[0]])
        files = [(train_dict[lst[0]][i], train_dict[lst[2]][i], train_dict[lst[4]][i]) for i in range(length)]
        
        train_files, vt_files = train_test_split(
            files, 
            test_size=(TVT_SPLIT[1] + TVT_SPLIT[2]) / 100, 
            random_state=42
        )
        valid_files, test_files = train_test_split(
            vt_files, 
            test_size=TVT_SPLIT[2] / (TVT_SPLIT[1] + TVT_SPLIT[2]), 
            random_state=42
        )
        
        try:
            os.mkdir("/kaggle/working/cifar100")
        except:
            print("sub-folder already created")
        
        np.savez('/kaggle/working/cifar100/train_files.npz', data=np.array(train_files, dtype=object))
        np.savez('/kaggle/working/cifar100/valid_files.npz', data=np.array(valid_files, dtype=object))
        np.savez('/kaggle/working/cifar100/test_files.npz', data=np.array(test_files, dtype=object))
    
    print("total classes: ", 100)
    classes = [i for i in range(100)]
    
    train_dataset = Cifar100Dataset(train_files, classes, 'train')
    valid_dataset = Cifar100Dataset(valid_files, classes, 'valid')
    test_dataset = Cifar100Dataset(test_files, classes, 'test')
    
    return train_dataset, valid_dataset, test_dataset
