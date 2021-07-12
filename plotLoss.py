import numpy as np
import matplotlib.pyplot as plt

ouiLossPath = "oui_train_loss.out"
stiLossPath = "sti_train_loss.out"
uirLossPath = "uir_train_loss.out"
discriminatorLossPath = "discriminator_train_loss.out"

ouiLoss = np.loadtxt(ouiLossPath, delimiter=',')
stiLoss = np.loadtxt(stiLossPath, delimiter=',')
uirLoss = np.loadtxt(uirLossPath, delimiter=',')
discriminatorLoss = np.loadtxt(discriminatorLossPath, delimiter=',')

plt.figure("OUI Loss")
plt.plot(ouiLoss)

plt.figure("STI Loss")
plt.plot(stiLoss)

plt.figure("UIR Loss")
plt.plot(uirLoss)

plt.figure("Discriminator Loss")
plt.plot(discriminatorLoss)

plt.show()

