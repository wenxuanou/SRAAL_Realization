import numpy as np
import matplotlib.pyplot as plt

ouiLossPath = "results/oui_train_loss_10%data_100epoch.out"
generatorLossPath = "results/generator_train_loss_10%data_100epoch.out"
discriminatorLossPath = "results/discriminator_train_loss_10%data_100epoch.out"
testAccPath = "results/test_accuracy_10%data_100epoch.out"

# ouiLossPath = "results/oui_train_loss.out"
# generatorLossPath = "results/generator_train_loss.out"
# discriminatorLossPath = "results/discriminator_train_loss.out"
# testAccPath = "results/test_accuracy.out"

ouiLoss = np.loadtxt(ouiLossPath, delimiter=',')
generatorLoss = np.loadtxt(generatorLossPath, delimiter=',')
discriminatorLoss = np.loadtxt(discriminatorLossPath, delimiter=',')
testAccuracy = np.loadtxt(testAccPath, delimiter=',')

plt.figure("OUI Loss")
plt.plot(ouiLoss)

plt.figure("Generator Loss")
plt.plot(generatorLoss)

plt.figure("Discriminator Loss")
plt.plot(discriminatorLoss)

plt.figure("Test Accuracy")
plt.plot(testAccuracy)

plt.show()

