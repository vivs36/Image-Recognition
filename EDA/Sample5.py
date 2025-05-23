#to show 5 specific sample images
from matplotlib import pyplot as plt
from keras.datasets import cifar10
import numpy as np

#to load dataset
(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()

class_labels = {
    0 : "airplane",
    1 : "automobile",
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck"
} 

np.bincount(training_labels.flatten(), minlength=10)
np.bincount(testing_labels.flatten(), minlength=10)
deer = training_images[training_labels.flatten() == 1][:5]
fig, axs = plt.subplots(1, 5, figsize=(15,3))
for ax, img in zip(axs, deer):
    ax.imshow(img)
    ax.axis("off")
fig.suptitle("Sample Automobile Images")
plt.show()
