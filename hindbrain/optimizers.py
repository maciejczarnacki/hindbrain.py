import numpy as np


class SGD:
    '''
    Stochastic gradient descent neural network weights/biases update algorithm with momentum.
    If momentum will be equal 0, it will be a classic stochastic gradient descent algorithm.
    '''
    def __init__(self, learning_rate = 0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, weights, biases, dw, db, *args):
        _, _, velocity_w, velocity_b, _, _  = args
        velocity_w = (1 - self.momentum) * dw + self.momentum * velocity_w
        velocity_b = (1 - self.momentum) * db + self.momentum * velocity_b
        weights -= self.learning_rate * velocity_w
        biases -= self.learning_rate * velocity_b

class RMSpopr:
    '''
    RMS propagation gradient descent neural network weights/biases update algorithm
    '''
    def __init__(self, learning_rate=0.001, beta=0.999, eps=10**(-7)):
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps

    def update(self, weights, biases, dw, db, *args):
        squared_grad_w, squared_grad_b, _, _, _, _ = args
        squared_grad_w = self.beta * squared_grad_w + (1 - self.beta) * (dw)**2
        squared_grad_b = self.beta * squared_grad_b + (1 - self.beta) * (db)**2
        weights -= self.learning_rate * dw / (squared_grad_w + self.eps)**0.5
        biases -= self.learning_rate * db / (squared_grad_b + self.eps)**0.5

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

    def update(self, weights, biases, dw, db, *args):
        squared_grad_w, squared_grad_b, velocity_w, velocity_b, squared_grad_w_max, squared_grad_b_max = args
        squared_grad_w = self.beta * squared_grad_w + (1 - self.beta) * (dw)**2
        squared_grad_b = self.beta * squared_grad_b + (1 - self.beta) * (db)**2
        velocity_w = (1 - self.momentum) * dw + self.momentum * velocity_w
        velocity_b = (1 - self.momentum) * db + self.momentum * velocity_b
        squared_grad_w_max = np.maximum(squared_grad_w_max, squared_grad_w)
        squared_grad_b_max = np.maximum(squared_grad_b_max, squared_grad_b)
        weights -= self.learning_rate * velocity_w / (np.sqrt(squared_grad_w_max) + self.eps)
        biases -= self.learning_rate * velocity_b / (np.sqrt(squared_grad_b_max) + self.eps)

"""     def update(self, weights, biases, dw, db, *args):
        # Original Adam from publication 
        # t += 1
            # squared_grad_w = self.beta * squared_grad_w + (1 - self.beta) * (dw.T @ dw)
            # squared_grad_b = self.beta * squared_grad_b + (1 - self.beta) * (db.T @ db)
            # velocity_w = self.momentum * velocity_w + (1 - self.momentum) * dw
            # velocity_b = self.momentum * velocity_b + (1 - self.momentum) * db
            # Sw_t = squered_grad_w / (1 - np.power(self.beta, t))
            # Sb_t = squered_grad_b / (1 - np.power(self.beta, t))
            # Vw_t = velocity_w / (1 - np.power(self.momentum,t))
            # Vb_t = velocity_b / (1 - np.power(self.momentum,t))
            # weights -= self.learning_rate * Vw_t / (np.sqrt(Sw_t) + self.eps)
            # biases -= self.learning_rate * Vb_t / (np.sqrt(Sb_t) + self.eps) """



