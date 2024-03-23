import numpy as np
import hindbrain as hb

my_model = hb.Model()

my_model.add_layer(hb.InputLayer(2))
my_model.add_layer(hb.LinearLayer(20), 'relu')
my_model.add_layer(hb.LinearLayer(20), 'relu')
my_model.add_layer(hb.LinearLayer(1), 'sigmoid')

my_model.build(loss='mse', learning_rate=0.15)

my_model.summary()

# xor gate inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

for i in range(600):
    for input, label in zip(inputs, labels):
        my_model.train(input, label, epochs=1)

# save model to file
# file_path = 'my_model_sigmoid'
# hb.save_model(file_path, my_model)

print('Input: ', inputs[0], 'Output:', my_model.predict(inputs[0]))
print('Input: ', inputs[1], 'Output:', my_model.predict(inputs[1]))
print('Input: ', inputs[2], 'Output:', my_model.predict(inputs[2]))
print('Input: ', inputs[3], 'Output:', my_model.predict(inputs[3]))
