import numpy as np

def mse(y_pred, y_true):
    mse_value = ((y_pred - y_true)**2).sum() / y_pred.size
    return mse_value

def d_mse(y_pred, y_true):
    d_mse_value = (y_pred - y_true)
    return d_mse_value

def mae(y_pred, y_true):
    mae_value = abs(y_pred - y_true).sum() / y_pred.size
    return mae_value

def d_mae(y_pred, y_true):
    if y_pred > y_true:
        d_mae_value = 1
    elif y_pred < y_true:
        d_mae_value = -1
    else:
        d_mae_value = 0
    return d_mae_value

def categorical_cross_entropy(y_true, y_pred):                   
    return -np.sum(y_true * np.log(y_pred + 10**-10))

def d_categorical_cross_entropy(y_true, y_pred):          
    return -y_true/(y_pred + 10**-10)

def binary_cross_entropy(y_true, y_pred):                   
    return -np.sum(y_true * np.log(y_pred + 10**-10)) - np.sum((1 - y_true) * np.log((1 - y_pred) + 10**-10))

def d_binary_cross_entropy(y_true, y_pred):          
    return -y_true/(y_pred + 10**-10)