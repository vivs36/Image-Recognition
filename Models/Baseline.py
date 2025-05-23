#to import libraries
import numpy as np
from keras.datasets import cifar10
import tensorflow as tf

#to load the dataset
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

# Normalize pixel values
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Convert labels from shape (n, 1) to (n,)
training_labels = training_labels.squeeze()
testing_labels = testing_labels.squeeze()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(60, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(40, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(160, activation="relu"),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(160, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(80, activation="relu"),
    tf.keras.layers.Dense(40, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(training_images, training_labels, epochs=14 , validation_data=(testing_images , testing_labels))

# Predict and evaluate manually
predictions = model.predict(testing_images)
predicted_labels = np.argmax(predictions, axis=1)
manual_accuracy = np.mean(predicted_labels == testing_labels)
print('Manual test accuracy:', manual_accuracy)

