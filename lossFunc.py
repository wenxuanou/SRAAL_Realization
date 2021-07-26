import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
#
# import numpy as np

def OUI_loss(output, labels):
    # Task model loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(output, labels)
    loss = torch.sum(loss) / loss.size(0)
    return loss


def STI_loss(mu, logVar, pred, labels):
    # only labeled data can provide loss for STI

    # KL divergence, assume normal distribution, Gaussian prior
    kl_div = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
    # classification loss
    criterion = nn.CrossEntropyLoss()
    class_loss = criterion(pred, labels)    # compare prediction and label

    loss = class_loss + kl_div
    return loss


def UIR_loss(mu, logVar, recons, imgs):
    # Apply twice for labeled and unlabeled data
    kl_div = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
    loss = F.mse_loss(recons, imgs, size_average=False) + kl_div

    return loss


def adversary_loss(labeled_preds, unlabeled_preds, lab_real_preds, unlab_real_preds):
    criterion = nn.BCELoss()
    loss = criterion(labeled_preds, lab_real_preds) + criterion(unlabeled_preds, unlab_real_preds)

    return loss

def discriminator_loss(labeled_preds, unlabeled_preds, lab_real_preds, unlab_fake_preds):
    criterion = nn.BCELoss()
    loss = criterion(labeled_preds, lab_real_preds) + criterion(unlabeled_preds, unlab_fake_preds)
    return loss
