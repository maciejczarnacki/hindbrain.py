import numpy as np

class InputLayer():
    def __init__(self, size, flatten=False, scale=None):
        self.size = size
        self.scale = scale
        self.flatten = flatten
        self.layer_name = 'input_layer'
        self.forward_activation_output = None
        self.forward_output = None
        self.forwart_input = None
        
    def prepare_data(self, x):
        if self.scale != None:
            self.input = x / self.scale
        else:
            self.input = x
        return self.input
    
    def forward(self, x):
        self.forwart_input = x
        if self.flatten:
            self.forward_output = x.flatten()
        else:
            self.forward_activation_output = x
            self.forward_output = x
        return self.forward_output
    

class LinearLayer:
    def __init__(self, size, trainable=True):
        self.size = size
        self.output_size = None
        self.weights = None
        self.biases = None
        self.learning_rate = None
        self.forward_output = None
        self.forward_activation_output = None
        self.forward_input = None
        self.trainable = trainable
        self.layer_name = 'linear_layer'
        self.x = None
        # momentum parameters
        self.vb = 0 # velocity of biases change
        self.vw = 0 # velocity of weights change
        self.sw_max = 0
        self.sb_max = 0
        # rmsprop parameters
        self.sb = 0 # exponentially weighted average of the squared gradients 
        self.sw = 0
        # adam parameters
        self.t = 0
    
    def forward(self, x):
        self.x = x
        self.forward_input = x
        self.forward_output = np.dot(x, self.weights) + self.biases
        return self.forward_output

# work in progress
class Dropout:
    def __init__(self, dropout_rate, trainable=True):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x):
        self.x = x
        self.forward_input = x
        self.mask = np.random.binomial([np.ones_like(self.x)],(1-self.dropout_rate))[0]  / (1-self.dropout_rate)
        self.forward_output = np.multiply(self.x, self.mask)
        self.forward_output /= (1.0 - self.dropout_rate)
        return self.forward_output
    
    def backward(self):
        pass


class ConvLayer2D:
    def __init__(self, kernel_size, stride=(1, 1), padding='valid'):
        self.input_size = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = None
    
    def forward(self, x):
        output = np.zeros(shape=((self.input_size[0] - self.kernel_size[0])/self.stride + 1, (self.input_size[0] - self.kernel_size[0])/self.stride + 1))
        for i in range(0, self.input_size[0] - self.kernel_size[0] + 1, self.stride):
            for j in range(0, self.input_size[0] - self.kernel_size[0] + 1, self.stride):
                step = x[j:j+3, i:i+3]
                output[i, j] = (step * self.kernel).sum()
            return output
    
    def backward(self):
        pass
    
class MaxPooling2D:
    def __init__(self, input_size, kernel_size):
        self.input_size = input_size
        self.kernel_size = kernel_size
        
    def foraward(self, x):
        pass
    
    def backward(self):
        pass