# Simple ResNet training for accuracy comparison

import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as sampler
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import random
from ResNet import resnet              # Use ResNet as task model
from tqdm import tqdm

import customDataset
from lossFunc import Resnet_loss

def cifar_transformer(isTrain):
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

def loadData(data_path, batch_size):
    # Load CIFAR10

    testset = datasets.CIFAR10(data_path, download=True,
                               transform=cifar_transformer(False), train=False)

    test_loader = data.DataLoader(testset, batch_size=batch_size,
                                 shuffle = True, num_workers = 2, drop_last=False)

    trainset = customDataset.CIFAR10(data_path)     # customized CIFAR10, added sample index info

    return test_loader, trainset

def labeledSetInit(num_train, M):
    all_indices = list(range(num_train))        # index for all samples

    # Randomly select labeled data
    random.shuffle(all_indices)
    labeled_indices = all_indices[:M]
    unlabeled_indices = all_indices[M:]

    return labeled_indices, unlabeled_indices

def updateDataloader(labeled_indices, BatchSize, train_set):
    # select part of dataset
    labeled_sampler = sampler.SubsetRandomSampler(labeled_indices)
    labeled_dataloader = data.DataLoader(train_set, sampler=labeled_sampler,
                                         batch_size=BatchSize,
                                         drop_last=True)  # labeled dataset, drop out not filed batch

    return labeled_dataloader

def extract_data(dataloader, labels=True):
    # make dataloader iterable, generator
    if labels:
        while True:
            for data in dataloader:
                img, label, id = data
                yield img, label, id
    else:
        while True:
            for data in dataloader:
                img, _, id = data
                yield img, id


if __name__=="__main__":
    print("Program start")

    # Random Seed
    random.seed("Wenxuan Ou")
    torch.manual_seed(999)

    # Parameters
    DataPath = "./data"                 # dataset directory
    OutPath = "./results"               # output log directory
    BatchSize = 128                     # batch size for training and testing
    ImgNum = 50000                      # CIFAR10 training set has 50000 samples in total
    Epochs = 50                         # training epochs (original: 100)
    M = ImgNum * 0.4                    # train set size (original: 100%)
    ClassNum = 10                       # CIFAR10: 10; CIFAR100: 100
    # ResNet Parameters
    LR = 0.1
    MILESTONES = [25, 35]
    MOMENTUM = 0.9
    WDECAY = 5e-4

    # Compute actual training iterations per epoch
    train_iterations = ImgNum // BatchSize

    # Device available
    ngpu = 1  # number of gpu available
    print("Using device: " + "cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # device = torch.device("cpu")        # test with cpu

    # Load data
    print("Load data")
    test_loader, train_set = loadData(DataPath, BatchSize)
    # CIFAT10 comes with label, randomly select a few for labeled set and put the rest in unlabeled set
    print("Initializing labeled set")
    labeled_indices, unlabeled_indices = labeledSetInit(ImgNum, int(M))  # Labeled set index initialization


    # Initialize ResNet
    resnet = resnet.ResNet18(num_classes=ClassNum).to(device)
    # ResNet optimizer and scheduler
    optim_resNet = optim.SGD(resnet.parameters(), lr=LR,
                             momentum=MOMENTUM, weight_decay=WDECAY)
    sched_resNet = lr_scheduler.MultiStepLR(optim_resNet, milestones=MILESTONES)  # ResNet scheduler

    # Test loss
    test_accuracy = [0]

    # Start training
    print("Training start")
    for epoch in range(Epochs):
        # Update dataloader
        labeled_dataloader = updateDataloader(labeled_indices, BatchSize, train_set)
        labeled_data = extract_data(labeled_dataloader)  # make iterable

        # set ResNet in train mode
        resnet.train()

        print("Epoch: " + str(epoch + 1) + " / " + str(Epochs))

        for iter_count in tqdm(range(train_iterations)):

            labeled_imgs, labels, labeled_batch_id = next(labeled_data)

            labeled_imgs = labeled_imgs.to(device)      # send data to training device
            labels = labels.to(device)

            # Train ResNet
            optim_resNet.zero_grad()
            pred_resnet, _ = resnet(labeled_imgs)
            resnet_loss = Resnet_loss(pred_resnet, labels)          # cross entropy loss
            resnet_loss.backward()
            optim_resNet.step()

        # update ResNet scheduler
        sched_resNet.step()

        # Test target model (OUI)
        correct = 0
        total = 0
        # Stop tracking gradient
        resnet.eval()
        with torch.no_grad():
            for test_sample in test_loader:
                test_inputs, test_labels = test_sample
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)

                pred_test, _ = resnet.forward(test_inputs)

                _, predicted = torch.max(pred_test.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        test_accuracy.append(correct / total)
        print("Data: " + str(M) + " / " + str(ImgNum))
        print("Accuracy of the network on the 10000 test images: %d %%" % (
                100 * correct / total))
        print("\n")

    test_accuracy_record = np.array(test_accuracy)
    np.savetxt("results/resnet_test_accuracy.out", test_accuracy_record, delimiter=",")