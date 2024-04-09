import hindbrain as hb
import numpy as np
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import struct
from array import array

# mnist dataset preparation
train_labels_path = 'examples/mnist/train-labels.idx1-ubyte'
train_images_path = 'examples/mnist/train-images.idx3-ubyte'

test_labels_path = 'examples/mnist/t10k-labels.idx1-ubyte'
test_images_path = 'examples/mnist/t10k-images.idx3-ubyte'

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
# train_images_flat = hb.tools.flatten(train_images)

train_images_scaled = train_images/255

# testing dataset

test_labels = load_data(test_labels_path)
test_images = load_image_data(test_images_path)

test_images_scaled = test_images/255


train_labels_one_hot = hb.tools.one_hot(train_labels, depth=10)


# neural network model build with hindbrain pakage

mnist_model = hb.Model()

mnist_model.add_layer(hb.InputLayer(784, flatten=True))
mnist_model.add_layer(hb.LinearLayer(128), activation='relu')
mnist_model.add_layer(hb.LinearLayer(128), activation='relu')
mnist_model.add_layer(hb.LinearLayer(128), activation='relu')
mnist_model.add_layer(hb.LinearLayer(10), activation='softmax')

mnist_model.summary()

mnist_model.build(loss='categorical_cross_entropy', optimizer='SGD', learning_rate=0.0001, momentum=0.9, beta=0.999)

av_loss = 0
total_loss = 0
m = 1
for epoch in range(3):
    for n, (data, label) in enumerate(zip(train_images_scaled, train_labels_one_hot)):
        mnist_model.train(data, label)
        total_loss = total_loss + mnist_model.loss_value
        av_loss = total_loss / m
        m += 1
        y_test = []
        if n%200 == 0:
            for test in test_images[:200]:
                y_test.append(np.argmax(mnist_model.predict(test)))
            acc = hb.accuracy(test_labels[:200], y_test)
            print(f'epoch: {epoch}, step: {n}, loss: {mnist_model.loss_value:.6f}, av_loss: {av_loss:.4f}, accuracy: {acc:.3f}')


# model accuracy evaluation on testing data
y_preds = []



c = 0
for z in range(10000):
    a = np.argmax(mnist_model.predict(test_images_scaled[z]))
    y_preds.append(a)

acc = hb.accuracy(test_labels, y_preds)
print('Total accuracy test: ', acc)
        
# print(f'Accuracy: {c/10000}')

lab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# conf = confusion_matrix(test_labels, y_pred=y_preds, labels=lab)

# print(conf)

hb.draw_confusion_matrix(test_labels, y_preds, lab, path='mnist_confusion_matrix.png', savefig=True)







