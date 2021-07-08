import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

import numpy as np

def OUI_loss(output, labels):
    # Loss for the classifier
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    return loss

def STI_loss(mu, logVar, pred, labels):
    # only labeled data can provide loss for STI
    # TODO: find out how label works here

    # KL divergence, assume normal distribution, Gaussian prior
    kl_div = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
    # classification loss
    criterion = nn.CrossEntropyLoss()
    class_loss = criterion(pred, labels)    # compare prediction and label

    loss = class_loss + kl_div
    return loss

def UIR_loss(mu, logVar, recons, imgs):
# def UIR_loss(mu_unlabeled, logVar_unlabeled, recon_unlabeled, imgs_unlabeled,
#              mu_labeled, logVar_labeled, recon_labeled, imgs_labeled):


    # Apply twice for labeled and unlabeled data

    # # Unlabeled loss
    # # KL divergence, assume normal distribution, Gaussian prior
    # kl_div_unlabeled = 0.5 * torch.sum(-1 - logVar_unlabeled + mu_unlabeled.pow(2) + logVar_unlabeled.exp())
    # # Binary cross entropy for difference between reconstruction difference
    # L_unlabeled = F.mse_loss(recon_unlabeled, imgs_unlabeled, size_average=False) + kl_div_unlabeled
    #
    # # Labeled loss
    # kl_div_labeled = 0.5 * torch.sum(-1 - logVar_labeled + mu_labeled.pow(2) + logVar_labeled.exp())
    # L_labeled = F.mse_loss(recon_labeled, imgs_labeled, size_average=False) + kl_div_labeled

    # loss = L_labeled + L_unlabeled

    kl_div = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
    loss = F.mse_loss(recons, imgs, size_average=False) + kl_div

    return loss

# def Discriminator_loss(uncertainty=1, pred=None, isLabeled=True):
#     criterion = nn.BCELoss()            # cross entropy loss, expect values in [0, 1]
#     # labeled: 1, unlabeled: 0
#     label_state = torch.zeros_like(pred)
#     if isLabeled:
#         label_state = torch.ones_like(pred)
#
#     loss = criterion(pred, label_state) + np.log(uncertainty)
#
#     return loss

# TODO: find a way to merge these two functions
def Discriminator_labeled_loss(pred):
    criterion = nn.BCELoss()
    loss = criterion(pred, torch.zeros_like(pred))
    return loss

def Discriminator_unlabeled_loss(uncertainty, pred):
    criterion = nn.BCELoss()
    label_state = torch.zeros_like(pred)

    loss = criterion(pred, label_state) + torch.mean(uncertainty)

    return loss