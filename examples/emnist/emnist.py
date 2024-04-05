import hindbrain as hb
import numpy as np
import struct
from array import array
import random

# mnist dataset preparation
train_labels_path = 'emnist-balanced-train-labels-idx1-ubyte'
train_images_path = 'emnist-balanced-train-images-idx3-ubyte'

test_labels_path = 'emnist-balanced-test-labels-idx1-ubyte'
test_images_path = 'emnist-balanced-test-images-idx3-ubyte'

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

print(len(test_images))


train_labels_one_hot = hb.tools.one_hot(train_labels, depth=47)


# neural network model with hindbrain pakage

emnist_model = hb.Model()

emnist_model.add_layer(hb.InputLayer(784, flatten=True))
emnist_model.add_layer(hb.LinearLayer(1024), activation='tanh')
emnist_model.add_layer(hb.LinearLayer(512), activation='tanh')
emnist_model.add_layer(hb.LinearLayer(256), activation='tanh')
emnist_model.add_layer(hb.LinearLayer(256), activation='tanh')
emnist_model.add_layer(hb.LinearLayer(256), activation='tanh')
emnist_model.add_layer(hb.LinearLayer(47), activation='softmax')

emnist_model.summary()

emnist_model.build(loss='categorical_cross_entropy', optimizer='SGD', learning_rate=0.085, momentum=0.9)
whole_data = list(zip(train_images_scaled, train_labels_one_hot))
for i in range(4):
    random.shuffle(whole_data)
    data_, labels_ = zip(*whole_data)
    for n, (data, label) in enumerate(zip(data_, labels_)):
        emnist_model.train(data, label, epochs=1)
        if n%200 == 0:
            print(f'epoch: {i}, step: {n}, loss: {emnist_model.loss_value}')


# model accuracy evaluation on testing data
y_preds = []

c = 0
for z in range(len(test_images)):
    a = np.argmax(emnist_model.predict(test_images_scaled[z]))
    y_preds.append(a)
    b = test_labels[z]
    if a == b:
        c += 1
        
print(f'Accuracy: {c/len(test_images)}')

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

hb.draw_confusion_matrix(test_labels, y_preds, classes=labels, path='emnist_confusion_matrix_SGD_tanh_1024.png', savefig=True)




