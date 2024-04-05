import numpy as np

# function for one hot labels convertion
def one_hot(x: np.ndarray, depth: int) -> np.ndarray:
  return np.take(np.eye(depth), x, axis=0)

# works in progress - (mini)batched dataset flattening
def flatten(X: np.ndarray) -> np.ndarray:
    """
    Flatten array in batches.

    Parameters
    ----------
    X: input numpy array.

    Retruns
    ----------
    Flattened X numpy array organized in batches like input array.
    """
    X_ = np.expand_dims(X, axis=1)
    return np.array([x.flatten() for x in X_])
