# Wenxuan Ou Summer 2021 Research Project, XuLab CMU
# Test trained model

import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as sampler
import torch.utils.data as data

import numpy as np
import random

from models import OUI

def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
        ])

# Load CIFAR10 test set
def loadCIFAR10(data_path, batch_size):
    testset = datasets.CIFAR10(data_path, download=True,
                               transform=cifar_transformer(), train=False)

    test_loader = data.DataLoader(testset, batch_size=batch_size,
                                  shuffle=True, num_workers=2, drop_last=False)

    return test_loader

if __name__ == "__main__":
    DataPath = "./data"         # dataset directory
    BatchSize = 4             # batch size for training and testing
    ModelPath = "results/oui_state_dict.pt"
    ClassNum = 10               # 10 classes in CIFAR10

    # Load data
    test_loader = loadCIFAR10(DataPath, BatchSize)

    # Load model
    model = OUI(channelNum=3, classNum=ClassNum, ngpu=1)        # This should match the saved model
    model.load_state_dict(torch.load(ModelPath))
    model.eval()

    # Predict output and get loss
    correct = 0
    total = 0
    # Stop tracking gradient
    with torch.no_grad():
        for data_sample in test_loader:
            inputs, labels = data_sample

            pred = model.forward(inputs)

            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))