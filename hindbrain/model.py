import numpy as np
import pickle
from .activations import relu, d_relu, sigmoid, d_sigmoid, linear, d_linear, softmax, d_softmax, tanh, d_tanh, elu, d_elu
from .losses import mse, d_mse, mae, d_mae, categorical_cross_entropy, d_categorical_cross_entropy, binary_cross_entropy, d_binary_cross_entropy

class Model:
    def __init__(self, name='Default_model', data_type='float32'):
        self.name = name
        self.model = []
        self.forward_passed = False
        self.weights = None
        self.biases = None
        self.loss = None
        self.data_type = data_type
        self.history = []
    
    def add_layer(self, layer, activation = 'None'):
        layer_number = len(self.model)
        self.model.append((layer_number, layer, activation))
    
    def build(self, loss, learning_rate=0.001, momentum=0.0):
        # generate weights and biases
        self.loss = loss
        self.learning_rate = learning_rate
        self.momentum = momentum
        sizes = []
        self.input_size = self.model[0][1].size
        for number, layer, activation in self.model:
            sizes.append(layer.size)
        self.biases = [np.zeros(shape=(y, 1)).reshape(1, y) for y in sizes[1:]]
        self.weights = [(np.random.uniform(-1, 1, size=(x, y))/np.sqrt(self.input_size))
                        for x, y in zip(sizes[:-1], sizes[1:])]
        for i, (number, layer, activation) in enumerate(self.model[1:]):
            layer.weights = self.weights[i]
            layer.biases = self.biases[i]
            layer.learning_rate = self.learning_rate
            layer.momentum = self.momentum
         
    def activation_function(self, activation, x):
        if activation == 'relu': output = relu(x)
        elif activation == 'elu': output = elu(x)
        elif activation == 'softmax': output = softmax(x)
        elif activation == 'sigmoid': output = sigmoid(x)
        elif activation == 'linear': output = linear(x)
        elif activation == 'tanh': output = tanh(x)
        else: 
            print('Unknown activation function')
            return None
        return output
    
    def activation_function_derivative(self, activation, x):
        if activation == 'relu': output = d_relu(x)
        elif activation == 'elu': output = d_elu(x)
        elif activation == 'softmax': output = d_softmax(x)
        elif activation == 'sigmoid': output = d_sigmoid(x)
        elif activation == 'linear': output = d_linear(x)
        elif activation == 'tanh': output = d_tanh(x)
        else: 
            print('Unknown activation function')
            return None
        return output
    
    def predict(self, x):
        x_ = np.copy(x)
        if self.model != None:
            for number, layer, activation in self.model:
                x_ = layer.forward(x_)
                if layer.layer_name !='input_layer':
                    x_ = self.activation_function(activation, x_)
                    layer.forward_activation_output = x_
            self.forward_passed = True
            return x_

    def train(self, x, labels, epochs=1):
        if self.model != None:
            for ep in range(epochs):
                preds = self.predict(x)
                # backpropagation steps layer by layer
                if self.loss == 'mse': 
                    loss = mse(preds, labels)
                    dx = d_mse(preds, labels)
                elif self.loss == 'mae': 
                    loss = mae(preds, labels)
                    dx = d_mae(preds, labels)
                elif self.loss == 'categorical_cross_entropy':
                    loss = categorical_cross_entropy(labels, preds)
                    dx = d_categorical_cross_entropy(labels, preds)
                elif self.loss == 'binary_cross_entropy':
                    loss = binary_cross_entropy(labels, preds)
                    dx = d_binary_cross_entropy(labels, preds)
                else: 
                    print('Unknown loss function')
                    return None
                print('loss', round(loss, 6))
                for i, (number, layer, activation) in enumerate(reversed(self.model[1:])):
                    if number == len(self.model) - 1:
                        if self.loss == 'categorical_cross_entropy':
                            delta = np.sum(dx * self.activation_function_derivative(activation, layer.forward_output), axis=1).reshape((-1, 1)).T
                        else:
                            dA = self.activation_function_derivative(activation, layer.forward_output)
                            delta = dx * dA
                        dw = np.dot(layer.forward_input.T, delta)
                        db = 1 * delta
                        layer.weights -= self.learning_rate * dw
                        layer.biases -= self.learning_rate * db
                    elif number != 1:
                        w_T = self.model[number+1][1].weights.T
                        dA = self.activation_function_derivative(activation, layer.forward_output)
                        dx = np.dot(delta, w_T)
                        delta = dx * dA
                        dw = np.dot(layer.forward_input.T, delta)
                        db = 1 * delta
                        layer.weights -= self.learning_rate * dw
                        layer.biases -= self.learning_rate * db
                
    def summary(self):
        print('Total number of layers: ', len(self.model))
        print('Model name: ', self.name)
        for number, layer, activation in self.model:
            print(f'Layer number: {number}, layer name: {layer.layer_name}, number of neurons: {layer.size}, activation: {activation}')
        print(f'Loss: {self.loss}')
        
def save_model(path, model) -> None:
    with open(path, 'wb') as file:
        pickle.dump(model, file)
            
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
            

