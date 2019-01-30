import numpy as np

def MSE(X, y, w):
    return np.mean((X @ w - y)**2)
