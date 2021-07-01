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
from models import OUI, STI, UIR, StateDiscriminator, weights_init
from lossFunc import OUI_loss, STI_loss, UIR_loss, Discriminator_labeled_loss, Discriminator_unlabeled_loss

# For debug only
import matplotlib.pyplot as plt

def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
        ])

def loadData(DatasetName, data_path, batch_size):
    if DatasetName == 'cifar10':
        testset = datasets.CIFAR10(data_path, download=True,
                                   transform=cifar_transformer(), train=False)

        test_loader = data.DataLoader(testset, batch_size=batch_size,
                                     shuffle = True, num_workers = 2, drop_last=False)

        trainset = datasets.CIFAR10(data_path, download=True,
                                   transform=cifar_transformer(), train=True)

        trainset_unlabeled = datasets.CIFAR10(data_path, download=True,
                                   transform=cifar_transformer(), train=True)

    elif DatasetName == 'cifar100':
        testset = datasets.CIFAR100(data_path, download=True,
                                   transform=cifar_transformer(), train=False)

        test_loader = data.DataLoader(testset, batch_size=batch_size,
                                     shuffle=True, num_workers=2, drop_last=False)

        trainset = datasets.CIFAR100(data_path, download=True,
                                    transform=cifar_transformer(), train=True)

        trainset_unlabeled = datasets.CIFAR100(data_path, download=True,
                                              transform=cifar_transformer(), train=True)

    # Select initial labeled set from trainset
    # Randomly sample K=ADDENDUM=1,000 data points
    NUM_TRAIN = 50000   # number of training images

    indices = list(range(NUM_TRAIN))
    random.shuffle(indices)
    unlabeled_set = indices      # indices for unlabeled data


    # train_loader = data.DataLoader(trainset, batch_size=batch_size,
    #                               sampler=sampler.SubsetRandomSampler(labeled_set), pin_memory=True)

    return testset, test_loader, trainset, trainset_unlabeled, unlabeled_set

def labeledSetInit(trainset, trainset_unlabeled, unlabeled_set, M, ZDim):
    # I = 10                  # I << M, M: labeled set size
    I = M

    # Randomly sample data to move from unlabeled to labeled set
    random.shuffle(unlabeled_set)
    labeled_set = unlabeled_set[:I]
    unlabeled_set = unlabeled_set[I:]

    # latent variables learned by unsupervised image reconstructor(UIR)
    # Z = computeLatent(trainset, unlabeled_set, ZDim)    # latent variables of all training data, both labeled and unlabeled
    # Z = np.zeros((ZDim, ZDim))                 # TODO: Z should be a numpy array here, ZDim * ZDim

    # while len(labeled_set) != M:
    #     # smallest distance between each Xl to all other Xu, dist is a numpy array
    #     dist = latentDist(labeled_set, unlabeled_set, Z)    # first col is sample index, second col is dist
    #
    #     # data point that the largest distance of any point to the point is minimum
    #     row = np.argmax(dist[:, 0])
    #     u = dist[row, 1]
    #
    #     labeled_set.append(u)   # TODO: check whether there are multiple u
    #     unlabeled_set.remove(u)

    # TODO: no longer select labeled data based on Z, simply randomly select

    return labeled_set, unlabeled_set

def computeLatent(trainset, unlabeled_set, ZDim):
    # Train with preset
    uir = UIR(channelNum=3, zDim=ZDim, ngpu=1).to(device)           # send model to GPU
    num_epochs = 1      # only train for a few epoch

    # Load in data loader, sample in labeled set using unlabeled index
    data_loader = data.DataLoader(trainset, batch_size=128,
                                  sampler=sampler.SubsetRandomSampler(unlabeled_set), pin_memory=True)


    # Optimizer
    optim_uir = optim.Adam(uir.parameters(), lr=5e-4)

    # Initialize weight
    uir.apply(weights_init)

    # Print the initialized network
    # print(uir)

    # Training
    # For each epoch
    print("Training start")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data_sample in enumerate(data_loader, 0):
            # print training info
            print("Epoch: " + str(epoch) + " / " + str(num_epochs)
                  + " Batch: " + str(i) + " / " + str(len(data_loader)))

            inputs, labels = data_sample         # CIFAR10 data comes with labeles, we only use input here
            inputs = inputs.to(device)

            # zero the parameter gradients
            optim_uir.zero_grad()

            recon, z, mu, logvar = uir.forward(inputs)
            # only use unlabeled part of the loss function
            kl_div = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())


            loss = F.mse_loss(recon, inputs, reduction='sum') + kl_div

            loss.backward()

            optim_uir.step()

            i += data_loader.batch_size

    # with torch.no_grad():
    #     input_sample = inputs[0, :, :, :]
    #     recon_sample = recon[0, :, :, :]
    #
    #     plt.figure(0)
    #     plt.imshow(input_sample.cpu().permute(1, 2, 0))
    #     plt.figure(1)
    #     plt.imshow(recon_sample.cpu().permute(1, 2, 0))
    #     plt.show()

    # Get latent variabes for all samples
    # TODO: compute latent for all data samples, z size: dataNum, 32
    data_loader = data.DataLoader(trainset, batch_size=128,
                                  sampler=sampler.SequentialSampler(unlabeled_set), pin_memory=True)
    Z = torch.empty((len(data_loader) * data_loader.batch_size, ZDim))
    # Compute latent for every unlabeled data
    print("Computing latent")
    for i, data_sample in enumerate(data_loader, 0):
        # print computing info
        print(" Batch: " + str(i) + " / " + str(len(data_loader)))

        inputs, labels = data_sample
        inputs = inputs.to(device)
        _, z, _, _ = uir.forward(inputs)
        # if on GPU, send back to CPU
        Z[i*data_loader.batch_size : i*data_loader.batch_size + z.shape[0], :] = z                 # push all latent tensor into list, z size: dataNum, 32
    return Z

def latentDist(labeled_set, unlabeled_set, Z):

    dist = torch.cdist()

    return dist


############################################
# Main
if __name__ == '__main__':
    print("Program start")

    # Random Seed
    random.seed("Wenxuan Ou")
    torch.manual_seed(999)

    # Parameters
    DatasetName = 'cifar10'     # name of dataset, use CIFAR10 or CIFAR100
    DataPath = './data'         # dataset directory
    OutPath = './results'       # output log directory
    LogName = 'accuracies.log'  # save final model performance
    BatchSize = 128             # batch size for training and testing
    # Epochs = 100                # training epochs
    Epochs = 10
    ZDim = 32                   # VAE latent dimension
    Beta = 1                    # VAE hyperparameter
    M = 1000                    # initial labeled set size
    ClassNum = 10               # CIFAR10: 10; CIFAR100: 100
    RelabelNum = 500            # number of samples to move to labeled set each epoch

    # Device available
    ngpu = 1    # number of gpu available
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # device = torch.device("cpu")        # test with cpu

    # Load data
    print("Load data")
    testset, test_loader, trainset, trainset_unlabeled, unlabeled_set = loadData(DatasetName, DataPath, BatchSize)
    print("Data Loaded")

    # Labeled set initialization
    # CIFAT10 comes with label, randomly select a few for labeled set and put the rest in unlabeled set
    print("Initializing labeled set")
    labeled_set, unlabeled_set = labeledSetInit(trainset, trainset_unlabeled, unlabeled_set, M, ZDim)

    # Initialize network
    uir = UIR(channelNum=3, zDim=ZDim, ngpu=1).to(device)
    sti = STI(channelNum=3, zDim=ZDim, classNum=ClassNum, ngpu=1).to(device)
    oui = OUI(channelNum=3, classNum=ClassNum, ngpu=1).to(device)           # OUI trains the target model
    discriminator = StateDiscriminator(ZDim).to(device)

    uir.apply(weights_init)
    sti.apply(weights_init)
    oui.apply(weights_init)
    discriminator.apply(weights_init)

    # Initialize optimizer
    optim_uir = optim.Adam(uir.parameters(), lr=5e-4)
    optim_sti = optim.Adam(sti.parameters(), lr=5e-4)
    optim_oui = optim.SGD(oui.parameters(), lr=5e-4)       # TODO: check whether Adam works for all
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)


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

        # For each batch, with labeled data
        # Train OUI(target model), UIR, STI, and discriminator
        for i, data_sample in enumerate(train_labeled_loader, 0):
            print("Epoch: " + str(epoch) + " / " + str(Epochs)
                  + "    Labeled Batch: " + str(i) + " / " + str(len(train_labeled_loader)))

            inputs, labels = data_sample        # Labels size: batch_size * 1, number as labels, need to decode to 0 and 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            # set gradient to zero
            optim_uir.zero_grad()
            optim_sti.zero_grad()
            optim_discriminator.zero_grad()

            # train OUI(target model), here is a simple classifier
            pred_oui = oui.forward(inputs)
            oui_loss = OUI_loss(pred_oui, labels)                       # TODO: check scale
            oui_loss.backward()
            optim_oui.step()

            # train UIR
            recons_uir, z_uir, mu_uir, logvar_uir = uir.forward(inputs)
            uir_loss = UIR_loss(mu_uir, logvar_uir, recons_uir, inputs)
            uir_loss.backward()
            optim_uir.step()

            # train STI
            pred_sti, z_sti, mu_sti, logvar_sti = sti.forward(inputs)
            sti_loss = STI_loss(mu_sti, logvar_sti, pred_sti, labels)           # TODO: check prediction and label error
            sti_loss.backward()
            optim_sti.step()

            # train discriminator
            z = torch.cat([z_sti, z_uir], dim=0)
            z = z.detach()                                                      # no need to track gradient here
            pred_discriminator = discriminator.forward(z)                       # discriminator tells whether data is labeled/unlabeled
            discriminator_loss = Discriminator_labeled_loss(pred_discriminator)
            discriminator_loss.backward(retain_graph=True)                      # TODO: fix backward problem
            optim_discriminator.step()

            print("OUI Loss: " + str(oui_loss.item())
                  + " UIR Loss: " + str(uir_loss.item())
                  + " STI Loss: " + str(sti_loss.item()))

            i += train_labeled_loader.batch_size


        # For each batch, with unlabeled data
        # UIR, discriminator
        for i, data_sample in enumerate(train_unlabeled_loader, 0):
            print("Epoch: " + str(epoch) + " / " + str(Epochs)
                  + "    Unlabeled Batch: " + str(i) + " / " + str(len(train_unlabeled_loader)))

            inputs, _ = data_sample             # will not use labels, regarded as unlabeled data
            inputs = inputs.to(device)

            # set gradient to zero
            optim_oui.zero_grad()
            optim_uir.zero_grad()
            optim_discriminator.zero_grad()

            # train UIR
            recons_uir, z_uir, mu_uir, logvar_uir = uir.forward(inputs)
            uir_loss = UIR_loss(mu_uir, logvar_uir, recons_uir, inputs)         # TODO: check whether UIR check is shared with labeled
            uir_loss.backward()
            optim_uir.step()

            # get latent from STI
            _, z_sti, _, _ = sti.forward(inputs)

            # get uncertainty score from OUI
            pred_oui = oui.forward(inputs)                  # pred_oui is possibility vector
            uncertainty = oui.getUncertainty(pred_oui)

            # train discriminator
            z = torch.cat([z_sti, z_uir], dim=0)
            z = z.detach()                                  # no need to track gradient for z
            uncertainty = torch.cat([uncertainty, uncertainty], dim=0)          # make same size as z
            uncertainty = uncertainty.detach()                            # no need to track gradient for uncertainty

            pred_discriminator = discriminator.forward(z)  # TODO: check sign
            discriminator_loss = Discriminator_unlabeled_loss(uncertainty, pred_discriminator)      # TODO: fix negative problem
            discriminator_loss.backward()
            optim_discriminator.step()

            print("UIR Loss: " + str(uir_loss.item())
                  + " Discriminator Loss: " + str(discriminator_loss.item()))

            i += train_unlabeled_loader.batch_size

        # Relabeling
        print("Relabeling")
        relabeling_loader = data.DataLoader(trainset_unlabeled, batch_size=BatchSize,
                                            sampler=sampler.SequentialSampler(unlabeled_set), pin_memory=True)

        # Access data sequentially, track indices
        relabeling_set = [[], []]
        for i, data_sample in enumerate(relabeling_loader, 0):
            print("Unlabeled pool: " + str(i) + " / " + str(len(relabeling_loader)))

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
                relabeling_set[0] = relabeling_set[0][:RelabelNum]
                relabeling_set[1] = relabeling_set[1][:RelabelNum]
            i += relabeling_loader.batch_size

        # TODO: check size of relabeling_set
        # Update labeled and unlabeled set, reload dataloader
        relabeling_set = [int(e) for e in relabeling_set[0]]                        # make sure all integer
        labeled_set += relabeling_set              # move to labeled set
        unlabeled_set = [e for e in unlabeled_set if e not in relabeling_set]       # remove from unlabeled set


    # Save model
    print("Save model")
    torch.save(oui.state_dict(), "oui_state_dict")
    torch.save(uir.state_dict(), "uir_state_dict")
    torch.save(sti.state_dict(), "sti_state_dict")
    torch.save(discriminator.state_dict(), "discriminator_state_dict")