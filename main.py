# Wenxuan Ou Summer 2021 Research Project, XuLab CMU
# Reference code: https://github.com/sinhasam/vaal,
#                 https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
# ResNet training parameter reference: https://github.com/kuangliu/pytorch-cifar


import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as sampler
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import random
from tqdm import tqdm                   # progress bar

from ResNet import resnet               # Use ResNet as task model
from models import Generator, StateDiscriminator, weights_init
from lossFunc import Resnet_loss, STI_loss, UIR_loss, adversary_loss, discriminator_loss
from utils import getUncertainty
import customDataset



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


def labeledSetInit(num_train, M, I=10, generator=None, trainset=None, randomInit=True):
    all_indices = list(range(num_train))        # index for all samples
    train_iterations = num_train // BatchSize

    if randomInit:
        # Randomly select labeled data
        random.shuffle(all_indices)
        labeled_indices = all_indices[:M]
        unlabeled_indices = all_indices[M:]
    else:
        # Select labeled data base on latent space distance
        # TODO: this process can be speed up
        random.shuffle(all_indices)
        labeled_indices = all_indices[:I]       # randomly select I samples, I << M
        unlabeled_indices = all_indices[I:]

        while len(labeled_indices) < M:
            print("\n")
            print("Adding to labeled set: " + str(len(labeled_indices)) + " / " + str(M))
            # update dataloader
            labeled_dataloader, unlabeled_dataloader = updateDataloader(labeled_indices, unlabeled_indices,
                                                                        1, BatchSize, trainset) # use smaller batch size: 1
            latent_l = []
            minDist_u = []
            unlabeled_id = []

            # z_l is latent space, compute distance
            for labeled_data in labeled_dataloader:
                labeled_imgs, _, labeled_batch_id = labeled_data
                labeled_imgs = labeled_imgs.to(device)
                with torch.no_grad():
                    _, _, z_l, mu_l, logvar_l = generator.forward(labeled_imgs)  # labeled data flow
                latent_l.extend(z_l)                                            # record latent of labeled data
            latent_l = torch.stack(latent_l, dim=1)

            for unlabeled_data in tqdm(unlabeled_dataloader):
                unlabeled_imgs, _, unlabeled_batch_id = unlabeled_data
                unlabeled_imgs = unlabeled_imgs.to(device)
                with torch.no_grad():
                    _, _, z_u, mu_u, logvar_u = generator.forward(unlabeled_imgs)  # unlabeled data flow

                    dist = torch.cdist(z_u, latent_l.T, 2)          # compute euclidean distance: B, len(labeled_indices)
                    minDist, _ = torch.min(dist, dim=1)             # minDist size: B, 1
                minDist_u.append(minDist)
                unlabeled_id.extend(unlabeled_batch_id)

            # add the one with max of minDist to label set
            minDist_u = torch.cat(minDist_u, dim=0)
            minDist_u = minDist_u.view(-1)
            _, id = torch.topk(minDist_u, 10)                    # select top 10 samples add to labeled set, to speed up
            id = id.cpu()
            id = np.asarray(unlabeled_id)[id]  # convert to numpy array

            labeled_indices = list(labeled_indices) + list(id)
            unlabeled_indices = np.setdiff1d(list(all_indices), labeled_indices)

        # free up memory
        del labeled_dataloader
        del unlabeled_dataloader
        del minDist_u
        del id

    return labeled_indices, unlabeled_indices


def updateDataloader(labeled_indices, unlabeled_indices, BatchSize_l, BatchSize_u, train_set):
    labeled_sampler = sampler.SubsetRandomSampler(labeled_indices)
    labeled_dataloader = data.DataLoader(train_set, sampler=labeled_sampler,
                                         batch_size=BatchSize_l,
                                         drop_last=True)                     # labeled dataset, drop out not filed batch

    unlabeled_sampler = sampler.SubsetRandomSampler(unlabeled_indices)
    unlabeled_dataloader = data.DataLoader(train_set, sampler=unlabeled_sampler,
                                           batch_size=BatchSize_u, drop_last=False)

    return labeled_dataloader, unlabeled_dataloader


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
    BatchSize = 128                                 # batch size for training and testing
    ImgNum = 50000                                  # CIFAR10 training set has 50000 samples in total
    Cycles = 4                                      # active learning cycles
    Epochs = 50                                     # ResNet training epochs
    ZDim = 32                                       # VAE latent dimension
    Beta = 1                                        # VAE hyperparameter
    M = 5000                                        # initial labeled set size (original: 10%, 5000)
    ClassNum = 10                                   # CIFAR10: 10; CIFAR100: 100
    RelabelNum = 5000                               # number of samples to relabel each AL cycle (make sure reach MaxLabelSize after Cycles)
    MaxLabelSize = 20000                            # maximum label set size (40% of total set, 20000)
    RandomInit = False                              # initial label set sampling method, if False, use UIR to initialize
    I = 10                                          # for labeled set initialization using UIR, I << M

    # ResNet Parameters
    LR = 0.1
    MILESTONES = [25, 35]                          # schedulers parameters follow ResNet18 training examples
    MOMENTUM = 0.9
    WDECAY = 5e-4


    # Compute actual training iterations per epoch
    train_iterations = ImgNum // BatchSize

    # Device available
    ngpu = 1    # number of gpu available
    global device
    print("Using device: " + "cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # device = torch.device("cpu")        # test with cpu


    # Initialize network
    generator = Generator(channelNum=3, zDim=ZDim, classNum=ClassNum).to(device)
    discriminator = StateDiscriminator(ZDim).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Initialize ResNet
    resnet = resnet.ResNet18(num_classes=ClassNum).to(device)


    # Load data
    print("Load data")
    test_loader, train_set = loadData(DataPath, BatchSize)
    # CIFAT10 comes with label, randomly select a few for labeled set and put the rest in unlabeled set
    print("Initializing labeled set")
    if RandomInit:
        labeled_indices, unlabeled_indices = labeledSetInit(ImgNum, int(M))  # Labeled set index initialization
    else:
        labeled_indices, unlabeled_indices = labeledSetInit(ImgNum, int(M),
                                                            I, generator, train_set,
                                                            randomInit=RandomInit)  # Labeled set index initialization

    # Test loss
    test_accuracy = [0]


    # Start training
    print("Training start")
    for cycles in range(Cycles):
        # Update dataloader
        labeled_dataloader, unlabeled_dataloader = updateDataloader(labeled_indices, unlabeled_indices,
                                                                    BatchSize, BatchSize, train_set)
        labeled_data = extract_data(labeled_dataloader)  # make iterable
        unlabeled_data = extract_data(unlabeled_dataloader, labels=False)

        # Reinitialize optimizer for each AL cycle
        optim_generator = optim.Adam(generator.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        # ResNet optimizer and scheduler
        optim_resNet = optim.SGD(resnet.parameters(), lr=LR,
                                 momentum=MOMENTUM, weight_decay=WDECAY)
        sched_resNet = lr_scheduler.MultiStepLR(optim_resNet, milestones=MILESTONES)  # ResNet scheduler



        # Network training epoch
        for epoch in range(Epochs):
            print("\n")
            print("AL cycles: " + str(cycles + 1) + " / " + str(Cycles)
                  + " Epoch: " + str(epoch) + " / " + str(Epochs))

            # set ResNet in train mode
            resnet.train()

            for iter_count in tqdm(range(train_iterations)):

                labeled_imgs, labels, _ = next(labeled_data)
                unlabeled_imgs, _ = next(unlabeled_data)

                labeled_imgs = labeled_imgs.to(device)      # send data to training device
                labels = labels.to(device)
                unlabeled_imgs = unlabeled_imgs.to(device)

                # Train ResNet
                optim_resNet.zero_grad()
                pred_resnet, _ = resnet(labeled_imgs)
                resnet_loss = Resnet_loss(pred_resnet, labels)          # cross entropy loss
                resnet_loss.backward()
                optim_resNet.step()


                # Train VAE with target model
                # VAE step, train UIR and STI (Generator)
                y_l, recon_l, z_l, mu_l, logvar_l = generator.forward(labeled_imgs)  # labeled data flow
                uir_l_loss = UIR_loss(mu_l, logvar_l, recon_l, labeled_imgs)
                sti_loss = STI_loss(mu_l, logvar_l, y_l, labels)  # labeled data provide loss for sti

                _, recon_u, z_u, mu_u, logvar_u = generator.forward(unlabeled_imgs)  # unlabeled data flow
                uir_u_loss = UIR_loss(mu_u, logvar_u, recon_u, unlabeled_imgs)

                labeled_preds = discriminator(
                    mu_l.detach())  # mu computer from z, discriminator learn from latent space
                unlabeled_preds = discriminator(mu_u.detach())

                # labeled is 1, unlabeled is 0
                lab_real_preds = torch.ones(labeled_imgs.size(0), 1)
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0), 1)  # -E[log( D(q_phi(z_u|x_u)) )]
                lab_real_preds = lab_real_preds.to(device)
                unlab_real_preds = unlab_real_preds.to(device)

                adv_loss = adversary_loss(labeled_preds, unlabeled_preds, lab_real_preds, unlab_real_preds)

                # L_G = lambda1 * L_uir + lambda2 * L_sti + lambda3 * adv_loss, total generator loss
                uir_loss = uir_l_loss + uir_u_loss
                total_vae_loss = uir_loss + sti_loss + adv_loss

                optim_generator.zero_grad()
                total_vae_loss.backward()
                optim_generator.step()

                # Train Discriminator
                with torch.no_grad():
                    # Get uncertainty score
                    pred_l_oui, _ = resnet(unlabeled_imgs)  # pred_oui is prediction, need to map to 0~1
                    pred_l_oui = F.softmax(pred_l_oui, dim=1)
                    uncertainty = getUncertainty(pred_l_oui, ClassNum)
                    uncertainty = torch.reshape(uncertainty, [uncertainty.size(0), 1])

                mu_l = mu_l.detach()
                mu_u = mu_u.detach()
                labeled_preds = discriminator(mu_l)
                unlabeled_preds = discriminator(mu_u)

                lab_real_preds = torch.ones(labeled_imgs.size(0), 1)
                unlab_fake_preds = torch.ones(unlabeled_imgs.size(0),
                                              1)  # relate to uncertainty here, -E[log(uncertainty - D(q_phi(z_u|x_u)) )]
                unlab_fake_preds = unlab_fake_preds.to(device)
                lab_real_preds = lab_real_preds.to(device)

                unlab_fake_preds = unlab_fake_preds - uncertainty

                dsc_loss = discriminator_loss(labeled_preds, unlabeled_preds, lab_real_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()



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
            print("\n")
            print("Current labeled set size: " + str(len(labeled_indices)) +
                  " Unlabeled set size: " + str(len(unlabeled_indices)))
            print("Accuracy of the network on the 10000 test images: %d %%" % (
                    100 * correct / total))
            print("\n")




        # Relabeling each active learning cycles
        if len(labeled_indices) <= MaxLabelSize:  # labeled set increase to 40% full dataset
            print("\n")
            print("Relabeling")
            all_preds = []
            all_indices = []

            for unlabeled_data in unlabeled_dataloader:
                imgs, _, indices = unlabeled_data
                imgs = imgs.to(device)
                # stop tracking gradient, get discriminator prediction
                with torch.no_grad():
                    _, _, _, mu, _ = generator.forward(imgs)
                    preds = discriminator.forward(mu)
                # record state values and indices
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # need to multiply by -1 to be able to use torch.topk
            all_preds *= -1

            # pick samples with top K state values to relabel
            _, relabel_indices = torch.topk(all_preds, int(RelabelNum))
            relabel_indices = relabel_indices.cpu()
            relabel_indices = np.asarray(all_indices)[relabel_indices]  # convert to numpy array

            labeled_indices = list(labeled_indices) + list(relabel_indices)
            unlabeled_indices = np.setdiff1d(list(all_indices), labeled_indices)



    # # Save model
    print("Save model")
    torch.save(resnet.state_dict(), "results/resnet_state_dict.pt")

    # Save loss values
    print("Save loss record")
    test_accuracy_record = np.array(test_accuracy)
    np.savetxt("results/test_accuracy.out", test_accuracy_record, delimiter=",")
