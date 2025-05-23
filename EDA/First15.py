#to show 15 images
from matplotlib import pyplot as plt
from keras.datasets import cifar10

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

fig = plt.figure(figsize=(12, 8))
columns = 5
rows = 3
for i in range(1, columns*rows +1):
    img = training_images[i] # get an image, defined as "img"
    fig.add_subplot(rows, columns, i) # create subplot (row index, col index, which number of plot)
    plt.title("Label:" + str(training_labels[i][0]) + ", Class:" + class_labels[training_labels[i][0]]) # plot the image, along with its label
    plt.imshow(img)
plt.show()