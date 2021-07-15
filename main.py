# Wenxuan Ou Summer 2021 Research Project, XuLab CMU
# Reference code: https://github.com/sinhasam/vaal,
#                 https://github.com/Mephisto405/Learning-Loss-for-Active-Learning

import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as sampler
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import random
from operator import itemgetter
from models import OUI, Generator, StateDiscriminator, weights_init
from lossFunc import OUI_loss, STI_loss, UIR_loss, Discriminator_labeled_loss, Discriminator_unlabeled_loss

# For debug only
import matplotlib.pyplot as plt

def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
        ])

def loadData(data_path, batch_size, num_train):
    # Load CIFAR10

    testset = datasets.CIFAR10(data_path, download=True,
                               transform=cifar_transformer(), train=False)

    test_loader = data.DataLoader(testset, batch_size=batch_size,
                                 shuffle = True, num_workers = 2, drop_last=False)

    trainset = datasets.CIFAR10(data_path, download=True,
                               transform=cifar_transformer(), train=True)

    trainset_unlabeled = datasets.CIFAR10(data_path, download=True,
                                   transform=cifar_transformer(), train=True)


    # Select initial uunlabeled set from trainset
    indices = list(range(num_train))            # number of training samples
    random.shuffle(indices)
    unlabeled_set = indices                     # indices for unlabeled data


    return test_loader, trainset, trainset_unlabeled, unlabeled_set

def labeledSetInit(unlabeled_set, M):
    # I = 10                  # I << M, M: labeled set size
    I = M

    # Randomly sample data to move from unlabeled to labeled set
    random.shuffle(unlabeled_set)
    labeled_set = unlabeled_set[:I]
    unlabeled_set = unlabeled_set[I:]

    return labeled_set, unlabeled_set




############################################
# Main
if __name__ == "__main__":
    print("Program start")

    # Random Seed
    random.seed("Wenxuan Ou")
    torch.manual_seed(999)

    # Parameters
    DataPath = "./data"                             # dataset directory
    OutPath = "./results"                           # output log directory
    LogName = "accuracies.log"                      # save final model performance
    BatchSize = 128                                 # batch size for training and testing
    NUM_TRAIN = 50000                               # CIFAR10 training set has 50000 samples in total
    # Epochs = 100                                  # training epochs
    Epochs = 100
    ZDim = 32                                       # VAE latent dimension
    Beta = 1                                        # VAE hyperparameter
    M = NUM_TRAIN * 0.1                             # initial labeled set size (the paper selects 10% of the entire set)
    ClassNum = 10                                   # CIFAR10: 10; CIFAR100: 100

    RelabelNum = NUM_TRAIN * 0.9 * 0.05             # number of samples to relabel each epoch (paper uses 5% of the unlabeled set, dynamically)

    # Device available
    ngpu = 1    # number of gpu available
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # device = torch.device("cpu")        # test with cpu

    # Load data
    print("Load data")
    test_loader, trainset, trainset_unlabeled, unlabeled_set = loadData(DataPath, BatchSize, NUM_TRAIN)
    print("Data Loaded")

    # Labeled set initialization
    # CIFAT10 comes with label, randomly select a few for labeled set and put the rest in unlabeled set
    print("Initializing labeled set")
    labeled_set, unlabeled_set = labeledSetInit(unlabeled_set, int(M))

    # Initialize network
    # TODO: set to training mode before training
    generator = Generator(channelNum=3, zDim=ZDim, classNum=ClassNum, ngpu=1).to(device)
    oui = OUI(channelNum=3, classNum=ClassNum, ngpu=1).to(device)           # OUI trains the target model
    discriminator = StateDiscriminator(ZDim).to(device)

    generator.apply(weights_init)
    oui.apply(weights_init)
    discriminator.apply(weights_init)

    # Initialize optimizer
    optim_generator = optim.Adam(generator.parameters(), lr=5e-4)
    optim_oui = optim.SGD(oui.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)   # Use SGD for target classifier
    # optim_oui = optim.Adam(oui.parameters(), lr=5e-4)                                 # Adam converges faster than SGD
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)


    # Tracking training loss
    oui_train_loss_record = [0]
    uir_train_loss_record = [0]
    sti_train_loss_record = [0]
    discriminator_train_loss_record = [0]
    # Test loss
    test_loss_record = [0]

    # Start training
    # For each epoch
    print("Training start")
    for epoch in range(Epochs):
        # Set up training dataloader, will be updated every iteration
        print("Labeled_set size: " + str(len(labeled_set)) + " Unlabeled_set size: " + str(len(unlabeled_set)))

        train_labeled_loader = data.DataLoader(trainset, batch_size=BatchSize,
                                               sampler=sampler.SubsetRandomSampler(labeled_set), pin_memory=True)
        train_unlabeled_loader = data.DataLoader(trainset_unlabeled, batch_size=BatchSize,
                                                 sampler=sampler.SubsetRandomSampler(unlabeled_set), pin_memory=True)

        # Train with labeled data
        oui_train_loss = 0.0
        uir_train_loss = 0.0
        sti_train_loss = 0.0
        for i, data_sample in enumerate(train_labeled_loader, 0):

            inputs, labels = data_sample        # Labels size: batch_size * 1, number as labels, need to decode to 0 and 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            # set gradient to zero
            optim_oui.zero_grad()
            optim_generator.zero_grad()
            optim_discriminator.zero_grad()

            # train OUI(target model), here is a simple classifier
            pred_oui = oui.forward(inputs)
            oui_loss = OUI_loss(pred_oui, labels)                       # TODO: check scale
            oui_loss.backward()
            optim_oui.step()

            # train generator
            pred, recon, z, mu, logvar = generator.forward(inputs)
            uir_loss = UIR_loss(mu, logvar, recon, inputs)
            sti_loss = STI_loss(mu, logvar, pred, labels)                       # labeled data can provide loss for sti
            uir_loss.backward(retain_graph=True)                                # TODO: check if need retain_graph
            sti_loss.backward()
            optim_generator.step()


            # train discriminator
            z = z.detach()                                                      # no need to track gradient for z
            pred_discriminator = discriminator.forward(z)                       # discriminator tells whether data is labeled/unlabeled
            discriminator_loss = Discriminator_labeled_loss(pred_discriminator) # ground truth for labeled sample is one
            discriminator_loss.backward()
            optim_discriminator.step()

            oui_train_loss += oui_loss.item()
            uir_train_loss += uir_loss.item()
            sti_train_loss += sti_loss.item()
            # record 4 loss every minibatch
            if i % (len(train_labeled_loader)/4) == (len(train_labeled_loader)/4) - 1:
                print("Epoch: " + str(epoch) + " / " + str(Epochs) +
                      "  Labeled Batch:" + str(i) + " / " + str(len(train_labeled_loader)))
                print("OUI Loss: " + str(oui_train_loss / (len(train_labeled_loader)/4))
                      + " UIR Loss: " + str(uir_train_loss / (len(train_labeled_loader)/4))
                      + " STI Loss: " + str(sti_train_loss / (len(train_labeled_loader)/4)) )

                # record loss
                oui_train_loss_record.append(oui_train_loss / (len(train_labeled_loader)/4))
                uir_train_loss_record.append(uir_train_loss / (len(train_labeled_loader)/4))
                sti_train_loss_record.append(sti_train_loss / (len(train_labeled_loader)/4))
                discriminator_train_loss_record.append(discriminator_train_loss_record[-1])   # No new values for discriminator loss

                oui_train_loss = 0.0
                uir_train_loss = 0.0
                sti_train_loss = 0.0


        # Train with unlabeled data
        uir_train_loss = 0.0
        discriminator_train_loss = 0.0
        for i, data_sample in enumerate(train_unlabeled_loader, 0):

            inputs, _ = data_sample             # will not use labels, regarded as unlabeled data
            inputs = inputs.to(device)

            # set gradient to zero
            optim_generator.zero_grad()
            optim_discriminator.zero_grad()

            # train generator
            _, recon, z, mu, logvar = generator.forward(inputs)
            uir_loss = UIR_loss(mu, logvar, recon, inputs)
            uir_loss.backward()                                                   # unlabeled, only update UIR weight
            optim_generator.step()

            # get uncertainty score from OUI
            pred_oui = oui.forward(inputs)                  # pred_oui is possibility vector
            uncertainty = oui.getUncertainty(pred_oui)

            # train discriminator
            z = z.detach()                                  # no need to track gradient for z
            uncertainty = torch.cat([uncertainty, uncertainty], dim=0)    # make same size as z
            uncertainty = uncertainty.detach()                            # no need to track gradient for uncertainty

            pred_discriminator = discriminator.forward(z)  # TODO: check sign
            discriminator_loss = Discriminator_unlabeled_loss(uncertainty, pred_discriminator) # ground truth for labeled sample is zero
            discriminator_loss.backward()
            optim_discriminator.step()

            uir_train_loss += uir_loss.item()
            discriminator_train_loss += discriminator_loss.item()
            # record 4 loss every minibatch
            if i % (len(train_unlabeled_loader) / 4) == (len(train_unlabeled_loader) / 4) - 1:
                print("Epoch: " + str(epoch) + " / " + str(Epochs) +
                      "  Unlabeled Batch:" + str(i) + " / " + str(len(train_unlabeled_loader)))
                print("UIR Loss: " + str(uir_train_loss / (len(train_unlabeled_loader) / 4))
                      + " Discriminator Loss: " + str(discriminator_train_loss / (len(train_unlabeled_loader) / 4)))

                # record loss
                oui_train_loss_record.append(oui_train_loss_record[-1])
                uir_train_loss_record.append(uir_train_loss / (len(train_unlabeled_loader) / 10))
                sti_train_loss_record.append(sti_train_loss_record[-1])
                discriminator_train_loss_record.append(discriminator_train_loss / (len(train_unlabeled_loader) / 10))

                uir_train_loss = 0.0
                discriminator_train_loss = 0.0



        # Relabeling, every epoch
        if len(labeled_set) <= 0.4 * NUM_TRAIN:             # Relabel until labeled set reaches 40% of total samples
            print("Relabeling")
            relabeling_loader = data.DataLoader(trainset_unlabeled, batch_size=BatchSize,
                                                sampler=sampler.SequentialSampler(unlabeled_set), pin_memory=True)

            # Access data sequentially, track indices
            relabeling_set = [[], []]
            for i, data_sample in enumerate(relabeling_loader, 0):
                # print("Unlabeled pool: " + str(i) + " / " + str(len(relabeling_loader)))

                inputs, _ = data_sample  # will not use labels, regarded as unlabeled data
                inputs = inputs.to(device)

                # get corresponding indices in dataset
                startId = i * relabeling_loader.batch_size
                endId = startId + list(inputs.shape)[0]     # get exact number of data in current batch
                indices = unlabeled_set[startId:endId]

                # get uncertainty score from OUI
                pred_oui = oui.forward(inputs)              # pred_oui is possibility vector
                uncertainty = oui.getUncertainty(pred_oui)
                uncertainty = uncertainty.tolist()

                # add to relabeling list
                relabeling_set[0] += indices
                relabeling_set[1] += uncertainty

                relabeling_set = np.array(relabeling_set)
                ind = np.argsort(relabeling_set[1, :])      # sort by uncertainty
                ind = np.flip(ind)                          # descending order
                relabeling_set = relabeling_set[:, ind]
                relabeling_set = relabeling_set.tolist()

                # keep top uncertainty data to relabel
                if len(relabeling_set[0]) > RelabelNum:
                    relabeling_set[0] = relabeling_set[0][:int(RelabelNum)]
                    relabeling_set[1] = relabeling_set[1][:int(RelabelNum)]
                i += relabeling_loader.batch_size

            # Update labeled and unlabeled set, reload dataloader
            relabeling_set = [int(e) for e in relabeling_set[0]]                        # convert to integer
            labeled_set += relabeling_set                                               # move to labeled set
            unlabeled_set = [e for e in unlabeled_set if e not in relabeling_set]       # remove from unlabeled set

            # Update number of samples to relabel
            RelabelNum = len(unlabeled_set) * 0.05


        # Test target model (OUI)
        # TODO: set to evaluation mode before testing
        correct = 0
        total = 0
        # Stop tracking gradient
        with torch.no_grad():
            for test_sample in test_loader:
                test_inputs, test_labels = test_sample
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)

                pred_test = oui.forward(test_inputs)

                _, predicted = torch.max(pred_test.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        test_loss_record.append((1 - correct / total))
        print("Accuracy of the network on the 10000 test images: %d %%" % (
                100 * correct / total))



    # Save model
    print("Save model")
    torch.save(oui.state_dict(), "results/oui_state_dict.pt")
    torch.save(generator.state_dict(), "results/generator_state_dict.pt")
    torch.save(discriminator.state_dict(), "results/discriminator_state_dict.pt")

    # Save loss values
    print("Save loss record")
    oui_train_loss_record = np.array(oui_train_loss_record)
    uir_train_loss_record = np.array(uir_train_loss_record)
    sti_train_loss_record = np.array(sti_train_loss_record)
    discriminator_train_loss_record = np.array(discriminator_train_loss_record)
    test_loss_record = np.array(test_loss_record)

    np.savetxt("results/oui_train_loss.out", oui_train_loss_record, delimiter=",")
    np.savetxt("results/uir_train_loss.out", uir_train_loss_record, delimiter=",")
    np.savetxt("results/sti_train_loss.out", sti_train_loss_record, delimiter=",")
    np.savetxt("results/discriminator_train_loss.out", discriminator_train_loss_record, delimiter=",")
    np.savetxt("results/test_loss.out", test_loss_record, delimiter=",")
