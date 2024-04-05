# import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np
import struct
from array import array

# mnist = tf.keras.datasets.mnist 
# (x_train , y_train) , (x_test , y_test) = mnist.load_data() 
# x_train = tf.keras.utils.normalize(x_train , axis = 1)
# x_test = tf.keras.utils.normalize(x_test , axis=1)

# y_train_oh = tf.one_hot(y_train, depth=10)
# y_test_oh = tf.one_hot(y_test, depth=10)

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
test_labels_one_hot = one_hot(test_labels, depth=10)

model = tf.keras.Sequential() 
model.add(tf.keras.layers.Flatten(input_shape = (28 , 28))) 
model.add(tf.keras.layers.Dense(256 , activation='relu'))
model.add(tf.keras.layers.Dense(256 , activation='relu'))
model.add(tf.keras.layers.Dense(256 , activation='relu'))
model.add(tf.keras.layers.Dense(10 , activation='softmax')) 

# model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy']) 
model.compile(tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), loss='categorical_crossentropy' , metrics=['accuracy'])

# history = model.fit(train_images_scaled, train_labels_one_hot, batch_size=1, epochs=1) 

for i in range(1):
    for data, label in zip(train_images_scaled, train_labels_one_hot):
        model.fit(np.array([data]), np.array([label]), epochs=1)

print(model.evaluate(test_images_scaled, test_labels_one_hot))
