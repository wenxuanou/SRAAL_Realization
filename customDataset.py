# This file modifies dataset and adds sample index information
# Borrow from VAAL code

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np

def cifar10_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
           transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
       ])

class CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)