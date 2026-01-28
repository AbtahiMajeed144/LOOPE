"""
LOOPE Datasets Module
"""
from .cifar100 import Cifar100Dataset, load_cifar100_data
from .imagenet100 import ImageNet100Dataset, load_imagenet100_data
from .transforms import RandomRotate90, RandomGamma
