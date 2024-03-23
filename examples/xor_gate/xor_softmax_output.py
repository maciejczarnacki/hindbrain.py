import numpy as np
import hindbrain as hb

my_model = hb.Model()

my_model.add_layer(hb.InputLayer(2))
my_model.add_layer(hb.LinearLayer(20), 'relu')
my_model.add_layer(hb.LinearLayer(20), 'relu')
my_model.add_layer(hb.LinearLayer(2), 'softmax')

my_model.build(loss='categorical_cross_entropy', learning_rate=0.075)

my_model.summary()

# xor gate inputs and outputs softmax
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels_softmax = np.array([[1, 0], [0, 1]])

# labels for sigmoid output
labels = np.array([[0], [1], [1], [0]])

new_training_set = []
training_data = list(zip(inputs, labels))
for k in training_data:
    if k[1] == [0]:
        new_training_set.append((k[0], labels_softmax[0]))
    else:
        new_training_set.append((k[0], labels_softmax[1]))
        
new_training_set_np = np.asarray(new_training_set)

for i in range(1000):
    for input, label in new_training_set_np:
        my_model.train(input, label, epochs=1)

# save model to file
# file_path = 'my_model_softmax'
# hb.save_model(file_path, my_model)

print(my_model.predict(inputs[0]))
print(my_model.predict(inputs[1]))
print(my_model.predict(inputs[2]))
print(my_model.predict(inputs[3]))
