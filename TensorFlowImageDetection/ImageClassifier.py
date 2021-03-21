import tensorflow as tf
import numpy as np
from tensorflow import keras

#matlab for output
import matplotlib.pyplot as plt

# already labeled dataset (70k, 60k training, 10k test)
fashion_mnist = keras.datasets.fashion_mnist

# load data from datset: 60k 28x28 images
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# display (for visualization purposes)
print(train_labels[0])
print(train_images[0])
plt.imshow(train_images[0], cmap='gray', vmin=0,vmax=255)
plt.show()

#neural net model construction
model = keras.Sequential([
    # input is 28x28 array, flatten will make it into a 1d vector of what dimensions:
    keras.layers.Flatten(input_shape =(28,28)),

    # hidden layer has 128 nodes, relu returns value of node, or 0 if negative. Great for speed (compared to a sigmoid function)
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    #output is 0-9, output layer returns/spits out maximum node probability of what piece of clothing it is
    keras.layers.Dense(units=10, activation=tf.nn.softmax)  
])

# compile the keras built neural net model to make it ready to be trained
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train!
model.fit(train_images, train_labels, epochs=5)

# Test!
test_loss = model.evaluate(test_images, test_labels)


#more visualization!
plt.imshow(test_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()
print("Expected Classification:", test_labels[0])



# Predict!
predictions = model.predict(test_images)
#visualize our predictions to the console!
print("Our prediction (probabilities) for first image:", predictions[0])

#this is our actual prediction, the index of max of the predictions[0]
print("Our actual prediction:", list(predictions[0]).index(max(predictions[0])))
