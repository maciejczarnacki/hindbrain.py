import numpy as np

# function for one hot labels convertion
def one_hot(x, depth: int):
  return np.take(np.eye(depth), x, axis=0)