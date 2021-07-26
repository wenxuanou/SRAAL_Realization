# Wenxuan Ou Summer 2021 Research Project, XuLab CMU
# Reference code: https://github.com/sinhasam/vaal,
#                 https://github.com/Mephisto405/Learning-Loss-for-Active-Learning

import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as sampler
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import random
from tqdm import tqdm                 # progress bar

from ResNet import resnet              # Use ResNet as task model
from models import Generator, StateDiscriminator, weights_init
from lossFunc import OUI_loss, STI_loss, UIR_loss, adversary_loss, discriminator_loss
from utils import getUncertainty
import customDataset



def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
        ])

def loadData(data_path, batch_size):
    # Load CIFAR10

    testset = datasets.CIFAR10(data_path, download=True,
                               transform=cifar_transformer(), train=False)

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

def updateDataloader(labeled_indices, unlabeled_indices, BatchSize, train_set):
    labeled_sampler = sampler.SubsetRandomSampler(labeled_indices)
    labeled_dataloader = data.DataLoader(train_set, sampler=labeled_sampler,
                                         batch_size=BatchSize,
                                         drop_last=True)  # labeled dataset, drop out not filed batch

    unlabeled_sampler = sampler.SubsetRandomSampler(unlabeled_indices)
    unlabeled_dataloader = data.DataLoader(train_set, sampler=unlabeled_sampler,
                                           batch_size=BatchSize, drop_last=False)

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
    ImgNum = 50000                               # CIFAR10 training set has 50000 samples in total
    Epochs = 200                                  # training epochs (original: 100)
    ZDim = 32                                       # VAE latent dimension
    Beta = 1                                        # VAE hyperparameter
    M = ImgNum * 0.1                             # initial labeled set size (original: 10%)
    ClassNum = 10                                   # CIFAR10: 10; CIFAR100: 100
    RelabelNum = ImgNum * 0.9 * 0.05             # number of samples to relabel each epoch (original: 5% of the unlabeled set, dynamically)
    MaxLabelSize = 0.4 * ImgNum                  # maximum label set size available
    # ResNet Parameters
    LR = 0.1
    MILESTONES = [160]
    EPOCHL = 120  # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model
    MOMENTUM = 0.9
    WDECAY = 5e-4


    # Compute actual training iterations per epoch
    train_iterations = ImgNum // BatchSize

    # Target model learning rate changes with overall iterations
    lr_change = train_iterations * Epochs // 4          # original: divided by 4

    # Device available
    ngpu = 1    # number of gpu available
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # device = torch.device("cpu")        # test with cpu

    # Load data
    print("Load data")
    test_loader, train_set = loadData(DataPath, BatchSize)
    # CIFAT10 comes with label, randomly select a few for labeled set and put the rest in unlabeled set
    print("Initializing labeled set")
    labeled_indices, unlabeled_indices = labeledSetInit(ImgNum, int(M))  # Labeled set index initialization

    # Initialize network
    generator = Generator(channelNum=3, zDim=ZDim, classNum=ClassNum, ngpu=1).to(device)
    discriminator = StateDiscriminator(ZDim).to(device)
    # Initialize ResNet
    resnet = resnet.ResNet18(num_classes=10).to(device)

    generator.apply(weights_init)
    # oui.apply(weights_init)
    discriminator.apply(weights_init)

    # Initialize optimizer
    optim_generator = optim.Adam(generator.parameters(), lr=5e-4)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

    # ResNet optimizer and scheduler
    optim_resNet = optim.SGD(resnet.parameters(), lr=LR,
                             momentum=MOMENTUM, weight_decay=WDECAY)
    sched_resNet = lr_scheduler.MultiStepLR(optim_resNet, milestones=MILESTONES)    # ResNet scheduler


    # Test loss
    test_accuracy = [0]


    # Start training
    print("Training start")
    for epoch in range(Epochs):
        # Update dataloader
        labeled_dataloader, unlabeled_dataloader = updateDataloader(labeled_indices, unlabeled_indices,
                                                                    BatchSize, train_set)
        labeled_data = extract_data(labeled_dataloader)                         # make iterable
        unlabeled_data = extract_data(unlabeled_dataloader, labels=False)

        # update ResNet scheduler
        sched_resNet.step()
        # set ResNet in train mode
        resnet.train()

        print("Epoch: " + str(epoch + 1) + " / " + str(Epochs))

        for iter_count in tqdm(range(train_iterations)):

            labeled_imgs, labels, labeled_batch_id = next(labeled_data)
            unlabeled_imgs, unlabeled_batch_id = next(unlabeled_data)

            labeled_imgs = labeled_imgs.to(device)      # send data to training device
            labels = labels.to(device)
            unlabeled_imgs = unlabeled_imgs.to(device)

            # Train ResNet
            optim_resNet.zero_grad()
            pred_resnet, _ = resnet(labeled_imgs)
            resnet_loss = OUI_loss(pred_resnet, labels)          # cross entropy loss
            resnet_loss.backward()
            optim_resNet.step()


            # VAE step, train UIR and STI (Generator)
            y_l, recon_l, z_l, mu_l, logvar_l = generator.forward(labeled_imgs)    # labeled data flow
            uir_l_loss = UIR_loss(mu_l, logvar_l, recon_l, labeled_imgs)
            sti_loss = STI_loss(mu_l, logvar_l, y_l, labels)                       # labeled data provide loss for sti

            _, recon_u, z_u, mu_u, logvar_u = generator.forward(unlabeled_imgs)    # unlabeled data flow
            uir_u_loss = UIR_loss(mu_u, logvar_u, recon_u, unlabeled_imgs)

            labeled_preds = discriminator(mu_l)             # mu computer from z, discriminator learn from latent space
            unlabeled_preds = discriminator(mu_u)

            # labeled is 1, unlabeled is 0
            lab_real_preds = torch.ones(labeled_imgs.size(0), 1)
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0), 1)    # -E[log( D(q_phi(z_u|x_u)) )]
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
                _, _, _, mu_l, _ = generator.forward(labeled_imgs)
                _, _, _, mu_u, _ = generator.forward(unlabeled_imgs)

                # Get uncertainty score
                pred_l_oui, _ = resnet(unlabeled_imgs)                # pred_oui is prediction, need to map to 0~1
                pred_l_oui = F.softmax(pred_l_oui, dim=1)
                uncertainty = getUncertainty(pred_l_oui, ClassNum)            # TODO: check uncertainty values
                uncertainty = torch.reshape(uncertainty, [uncertainty.size(0), 1])
                uncertainty = uncertainty.cpu()

            labeled_preds = discriminator(mu_l)
            unlabeled_preds = discriminator(mu_u)

            lab_real_preds = torch.ones(labeled_imgs.size(0), 1)
            unlab_fake_preds = torch.ones(unlabeled_imgs.size(0), 1)       # relate to uncertainty here, -E[log(uncertainty - D(q_phi(z_u|x_u)) )]
            unlab_fake_preds = unlab_fake_preds - uncertainty               # TODO: check if this is correct
            lab_real_preds = lab_real_preds.to(device)
            unlab_fake_preds = unlab_fake_preds.to(device)

            dsc_loss = discriminator_loss(labeled_preds, unlabeled_preds, lab_real_preds, unlab_fake_preds)

            optim_discriminator.zero_grad()
            dsc_loss.backward()
            optim_discriminator.step()


        # Relabeling each epoch
        if len(labeled_indices) <= MaxLabelSize:               # labeled set increase to 40% full dataset
            print("Relabeling")
            all_preds = []
            all_indices = []

            for imgs, _, indices in unlabeled_dataloader:
                imgs = imgs.to(device)
                # stop tracking gradient, get discriminator prediction
                with torch.no_grad():
                    _, _, _, mu, _ = generator.forward(imgs)
                    preds = discriminator.forward(mu)
                # record state values and indices
                preds = preds.cpu().data
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # need to multiply by -1 to be able to use torch.topk
            all_preds *= -1

            # pick samples with top K state values to relabel
            _, relabel_indices = torch.topk(all_preds, int(RelabelNum))
            relabel_indices = np.asarray(all_indices)[relabel_indices]          # convert to numpy array

            labeled_indices = list(labeled_indices) + list(relabel_indices)
            unlabeled_indices = np.setdiff1d(list(all_indices), labeled_indices)

            # Update number of images to relabel
            RelabelNum = len(unlabeled_indices) * 0.05


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

                pred_test,_ = resnet.forward(test_inputs)

                _, predicted = torch.max(pred_test.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        test_accuracy.append(correct / total)
        print("Current labeled set size: " + str(len(labeled_indices)) +
              " Unlabeled set size: " + str(len(unlabeled_indices)))
        print("Accuracy of the network on the 10000 test images: %d %%" % (
                100 * correct / total))
        print("\n")



    # # Save model
    print("Save model")
    torch.save(resnet.state_dict(), "results/resnet_state_dict.pt")

    # Save loss values
    print("Save loss record")
    test_accuracy_record = np.array(test_accuracy)
    np.savetxt("results/test_accuracy.out", test_accuracy_record, delimiter=",")
