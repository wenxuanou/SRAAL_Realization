import numpy as np
import matplotlib.pyplot as plt

testAccPath = "results/test_accuracy.out"

resnetAccPath = "results/resnet_test_accuracy.out"


testAccuracy = np.loadtxt(testAccPath, delimiter=',')
resNetAccuracy = np.loadtxt(resnetAccPath, delimiter=',')

print("max test acc: " + str(np.amax(testAccuracy)))
print("max ResNet test acc: " + str(np.amax(resNetAccuracy)))


plt.figure("Test Accuracy")
plt.plot(testAccuracy)

plt.figure("ResNet Accuracy")
plt.plot(resNetAccuracy)

plt.show()

