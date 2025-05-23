#to show 5 sample images of each class
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

for i in range(10):
  obj = training_images[training_labels.flatten() == i][:5]
  fig, axs = plt.subplots(1,5, figsize=(15,3))
  for ax, img in zip(axs, obj):
      ax.imshow(img)
      ax.axis("off")
  fig.suptitle("Sample "+class_labels[i] +" Images")
  plt.show()