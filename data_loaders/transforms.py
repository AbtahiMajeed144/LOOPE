"""
Custom Image Transforms for Data Augmentation
"""
import random
import torchvision.transforms as T


class RandomRotate90:
    """Randomly rotate image by 0, 90, 180, or 270 degrees."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([0, 90, 180, 270])
            return T.functional.rotate(img, angle)
        return img


class RandomGamma:
    """Randomly adjust gamma of the image."""
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        self.gamma_range = gamma_range
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            return T.functional.adjust_gamma(img, gamma)
        return img
