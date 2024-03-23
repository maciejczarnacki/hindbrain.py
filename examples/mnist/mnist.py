import hindbrain as hb
import numpy as np
import struct
from array import array

# mnist dataset preparation
train_labels_path = 'train-labels.idx1-ubyte'
train_images_path = 'train-images.idx3-ubyte'

test_labels_path = 't10k-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'

# function for loading labels data from file
def load_data(filepath):
    with open(file=filepath, mode='rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())  
    return np.asarray(labels)

# function for loading images from file
def load_image_data(filepath):
    with open(file=filepath, mode='rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img      
    return np.asarray(images)

# training dataset
train_labels = load_data(train_labels_path)
train_images = load_image_data(train_images_path)

train_images_scaled = train_images/255

# testing dataset

test_labels = load_data(test_labels_path)
test_images = load_image_data(test_images_path)

test_images_scaled = test_images/255

# function for one hot labels convertion
def one_hot(x, depth: int):
  return np.take(np.eye(depth), x, axis=0)

train_labels_one_hot = one_hot(train_labels, depth=10)


# neural network model with hindbrain pakage

mnist_model = hb.Model()

mnist_model.add_layer(hb.InputLayer(784, flatten=True))
mnist_model.add_layer(hb.LinearLayer(300), activation='tanh')
mnist_model.add_layer(hb.LinearLayer(100), activation='tanh')
mnist_model.add_layer(hb.LinearLayer(10), activation='softmax')

mnist_model.summary()

mnist_model.build(loss='categorical_cross_entropy', learning_rate=0.001)

for i in range(2):
    for data, label in zip(train_images_scaled, train_labels_one_hot):
        mnist_model.train(data, label, epochs=1)


# model accuracy evaluation on testing data

c = 0
for z in range(10000):
    a = np.argmax(mnist_model.predict(test_images_scaled[z]))
    b = test_labels[z]
    if a == b:
        c += 1
        
print(f'Accuracy: {c/10000}')




