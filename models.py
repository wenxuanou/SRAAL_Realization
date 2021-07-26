import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.init as init
import torch.cuda
# import torch.backends.cudnn as cudnn


import numpy as np

# Tensor resize
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


# Generator, including STI and UIR
class Generator(nn.Module):
    def __init__(self, channelNum=3, featureDim=32 * 20 * 20, zDim=256, classNum=10, ngpu=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.channelNum = channelNum
        self.zDim = zDim
        self.featureDim = featureDim            # TODO: may not be using
        self.classNum = classNum                # TODO: may not be using

        self.cnn = nn.Sequential(
            nn.Conv2d(channelNum, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32; stride=2, each side decrease by 4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 2 * 2)),  # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024 * 2 * 2, zDim)  # B, z_dim
        self.fc_logvar = nn.Linear(1024 * 2 * 2, zDim)  # B, z_dim

        # STI decoder is a classifier for classification, on CIFAR10 dataset
        self.sti = nn.Sequential(
            nn.Linear(zDim, 1024 * 4 * 4),  # B, 1024*8*8
            View((-1, 1024, 4, 4)),  # B, 1024,  8,  8
            nn.Flatten(),  # Flatten all dimension except batch
            nn.Linear(1024 * 4 * 4, 120),   # TODO: tune layer dimension
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, classNum),        # For CIFAR10, classNum is 10
        )

        # UIR decoder reconstruct image from latent variables
        self.uir = nn.Sequential(
            nn.Linear(zDim, 1024 * 4 * 4),  # B, 1024*8*8
            View((-1, 1024, 4, 4)),  # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, channelNum, 1),  # B,   nc, 64, 64
        )

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def forward(self, x):
        z = self.cnn(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)     # latent variables, rich representation
        pred = self.sti(z)                      # STI output predictions
        recon = self.uir(z)                     # UIR output reconstructions
        return pred, recon, z, mu, logvar






# Label state discriminator
class StateDiscriminator(nn.Module):
    # Grab from Pytorch DCGAN example
    # Discriminator learns to differentiate labeled/unlabeled data
    def __init__(self, z_dim=10):
        super(StateDiscriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)      # output a single value for state loss, model state of samples


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)





###############################################################
# example VAE implementation
class VAE(nn.Module):
    def __init__(self,channelNum=3, featureDim=32*20*20, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(channelNum, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, channelNum, 5)

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32 * 20 * 20)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar
