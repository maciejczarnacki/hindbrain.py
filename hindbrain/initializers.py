# weights and biases initializers

import numpy as np

def he_normal(shape: tuple) -> np.ndarray:
    """
    Kaiming He et al., Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,
    https://doi.org/10.48550/arXiv.1502.01852
    """
    x, y = shape
    std = np.sqrt(2/y)
    weights_matrix = np.random.normal(0, std, size=shape)
    return weights_matrix

def he_uniform(shape: tuple) -> np.ndarray:
    """
    Kaiming He et al., Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,
    https://doi.org/10.48550/arXiv.1502.01852
    """
    x, y = shape
    d = np.sqrt(6/y)
    weights_matrix = np.random.uniform(-d, d, size=shape)
    return weights_matrix

def glorot_normal(shape: tuple) -> np.ndarray:
    """
    Xavier Glorot, Yoshuda Bengio, Understanding the difficulty of training deep feedforward neural networks,
    Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, PMLR 9:249-256, 2010.
    https://proceedings.mlr.press/v9/glorot10a.html
    """
    x, y = shape
    std = np.sqrt(2/(x + y))
    weights_matrix = np.random.normal(0, std, size=shape)
    return weights_matrix

def glorot_uniform(shape: tuple) -> np.ndarray:
    """
    Xavier Glorot, Yoshuda Bengio, Understanding the difficulty of training deep feedforward neural networks,
    Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, PMLR 9:249-256, 2010.
    https://proceedings.mlr.press/v9/glorot10a.html
    """
    x, y = shape
    d = np.sqrt(6/(x + y))
    weights_matrix = np.random.uniform(-d, d, size=shape)
    return weights_matrix

def zeros(shape: tuple) -> np.ndarray:
    weights_matrix = np.zeros(shape=shape)
    return weights_matrix

def weights_init(shape: tuple, initializer='he_normal') -> np.ndarray:
    if initializer == 'he_normal': return he_normal(shape=shape)
    elif initializer == 'he_uniform': return he_uniform(shape=shape)
    elif initializer == 'glorot_normal': return he_uniform(shape=shape)
    elif initializer == 'glorot_uniform': return he_uniform(shape=shape)
    else:
        raise ValueError('Unknown weights initializer.')

def biases_init(shape: tuple, initializer='zeros') -> np.ndarray:
    if initializer == 'zeros': return zeros(shape=shape)
    else:
        raise ValueError('Unknown biases initializer!')