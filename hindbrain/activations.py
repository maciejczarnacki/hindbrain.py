import numpy as np

# forward activation functions

def relu(x: np.ndarray) -> np.ndarray:
    x_ = np.copy(x)
    x_[x_<=0] = 0
    return x_

def elu(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    x_ = np.copy(x)
    out_ = np.where(x_>0, x_, alpha*(np.exp(x_)-1))
    return out_
    
def softmax(x: np.ndarray) -> np.ndarray:
    A = np.max(x)
    x_ = np.exp(x-A) / np.exp(x-A).sum()
    return x_

# Softamx for minibatches
def softmax_b(x: np.ndarray) -> np.ndarray:
    x_ = np.asarray([softmax(a) for a in x])
    return x_

def sigmoid(x: np.ndarray) -> np.ndarray:
    x_ = 1 / (1 + np.exp(-x))
    return x_

def linear(x: np.ndarray) -> np.ndarray:
    return x

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

# backward activation functions (derivatives)

def d_relu(x: np.ndarray) -> np.ndarray:
    x_ = np.copy(x)
    x_[x_<=0] = 0
    x_[x_>0] = 1
    return x_

def d_elu(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    x_ = np.copy(x)
    out_ = np.where(x_>0, np.ones_like(x_), alpha*np.exp(x_))
    return out_

def d_softmax(x: np.ndarray) -> np.ndarray:
    first_step = softmax(x).reshape(-1,1)
    x_ = first_step * (np.identity(first_step.size) - first_step.transpose())
    return x_

# Softmax derivation for minibatches
def d_softmax_b(x: np.ndarray) -> np.ndarray:
    x_ = np.asarray([[d_softmax(a) for a in x]])
    return x_

def d_sigmoid(x: np.ndarray) -> np.ndarray:
    x_ = sigmoid(x) * (1 - sigmoid(x))
    return x_

def d_linear(x: np.ndarray) -> np.ndarray:
    x_ = np.copy(x)
    x_[x_<=0] = 1
    x_[x_>0] = 1
    return x_

def d_tanh(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x)**2