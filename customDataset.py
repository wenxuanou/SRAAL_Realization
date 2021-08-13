# This file modifies dataset and adds sample index information
# Borrow from VAAL code

from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

def cifar10_transformer(isTrain):
    if isTrain:
        # For train set
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    else:
        # For test set
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

class CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer(True))

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)