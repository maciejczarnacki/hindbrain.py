import numpy as np

# forward activation functions

def relu(x):
    x_ = np.copy(x)
    x_[x_<=0] = 0
    return x_

def elu(x, alpha = 0.1):
    x_ = np.copy(x)
    out_ = np.maximum(alpha*(np.exp(x_)-1), x_)
    return out_
    
def softmax(x):
    x_ = np.exp(x) / np.exp(x).sum()
    return x_

def sigmoid(x):
    x_ = 1 / (1 + np.exp(-x))
    return x_

def linear(x):
    return x

def tanh(x):
    return np.tanh(x)

# backward activation functions

def d_relu(x):
    x_ = np.copy(x)
    x_[x_<=0] = 0
    x_[x_>0] = 1
    return x_

def d_elu(x, alpha=0.2):
    x_ = np.copy(x)
    out_ = np.maximum(alpha*np.exp(x_), 1)
    return out_

def d_softmax(x):
    first_step = softmax(x).reshape(-1,1)
    x_ = first_step * (np.identity(first_step.size) - first_step.transpose())
    return x_

def d_sigmoid(x):
    x_ = sigmoid(x) * (1 - sigmoid(x))
    return x_

def d_linear(x):
    x_ = np.copy(x)
    x_[x_<=0] = 1
    x_[x_>0] = 1
    return x_

def d_tanh(x):
    return 1 - np.tanh(x)**2