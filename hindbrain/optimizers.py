import numpy as np


class SGD:
    '''
    Stochastic gradient descent neural network weights/biases update algorithm with momentum.
    If momentum will be equal 0, it will be a classic stochastic gradient descent algorithm.
    '''
    def __init__(self, learning_rate = 0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def update(self, layer, dw, db, *args):
        layer.vw = (1 - self.momentum) * dw + self.momentum * layer.vw
        layer.vb = (1 - self.momentum) * db + self.momentum * layer.vb
        layer.weights -= self.learning_rate * layer.vw
        layer.biases -= self.learning_rate * layer.vb

"""     # SGD algorithm from Tensorflow
        layer.vw = self.momentum * layer.vw - self.learning_rate * dw
        layer.weights = layer.weights + layer.vw
        layer.vb = self.momentum * layer.vb - self.learning_rate * db
        layer.biases = layer.biases + layer.vb """


class RMSpopr:
    '''
    RMS propagation gradient descent neural network weights/biases update algorithm
    '''
    def __init__(self, learning_rate=0.001, beta=0.999, eps=10**(-7)):
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps

    def update(self, layer, dw, db, *args):
        layer.sw = self.beta * layer.sw + (1 - self.beta) * (dw)**2
        layer.sb = self.beta * layer.sb + (1 - self.beta) * (db)**2
        layer.weights -= self.learning_rate * dw / (layer.sw + self.eps)**0.5
        layer.biases -= self.learning_rate * db / (layer.sb + self.eps)**0.5


class Amsgrad:
    '''
    Variant of ADAM gradient descent neural network weights/biases update algorithm.
    AMSGrad described in - S. J. Reddi, S. Kale, S. Kumar, On the Convergence of Adam and Beyond, 2019,
    https://doi.org/10.48550/arXiv.1904.09237
    '''
    def __init__(self, learning_rate=0.001, momentum=0.9, beta=0.999, eps=10**(-7)):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.eps = eps

    def update(self, layer, dw, db, *args):
        layer.sw = self.beta * layer.sw + (1 - self.beta) * (dw)**2
        layer.sb = self.beta * layer.sb + (1 - self.beta) * (db)**2
        layer.vw = (1 - self.momentum) * dw + self.momentum * layer.vw
        layer.vb = (1 - self.momentum) * db + self.momentum * layer.vb
        layer.sw_max = np.maximum(layer.sw_max, layer.sw)
        layer.sb_max = np.maximum(layer.sb_max, layer.sb)
        layer.weights -= self.learning_rate * layer.vw / (np.sqrt(layer.sw_max) + self.eps)
        layer.biases -= self.learning_rate * layer.vb / (np.sqrt(layer.sb_max) + self.eps)


class Adam:
    '''
    Variant of ADAM gradient descent neural network weights/biases update algorithm.
    AMSGrad described in - S. J. Reddi, S. Kale, S. Kumar, On the Convergence of Adam and Beyond, 2019,
    https://doi.org/10.48550/arXiv.1904.09237
    '''
    def __init__(self, learning_rate=0.001, momentum=0.9, beta=0.999, eps=10**(-7)):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.eps = eps

    def update(self, layer, dw, db, *args):
        # Original Adam from publication 
        layer.t += 1
        layer.sw = self.beta * layer.sw + (1 - self.beta) * dw ** 2
        layer.sb = self.beta * layer.sb + (1 - self.beta) * db ** 2
        layer.vw = self.momentum * layer.vw + (1 - self.momentum) * dw
        layer.vb = self.momentum * layer.vb + (1 - self.momentum) * db
        Sw_t = layer.sw / (1 - np.power(self.beta, layer.t))
        Sb_t = layer.sb/ (1 - np.power(self.beta, layer.t))
        Vw_t = layer.vw / (1 - np.power(self.momentum,layer.t))
        Vb_t = layer.vb / (1 - np.power(self.momentum,layer.t))
        layer.weights -= self.learning_rate * Vw_t / (np.sqrt(Sw_t) + self.eps)
        layer.biases -= self.learning_rate * Vb_t / (np.sqrt(Sb_t) + self.eps)



