import numpy as np
from GD import gradient_descent

def Linear_regression(X, y, method = 0, w_0 = None, alpha_0 = 1, b = 1, eps = 1e-06):

    # local variables
    p = X.shape[1]
    optim_w = np.zeros(p)

    # computes the optimal weights using the closed form solution
    if(method == 0):
        X_T = X.T
        b = X_T @ y
        A = X_T @ X
        
        optim_w = np.linalg.solve(A,b)

        return optim_w

    # computes the optimal weights using gradient descent
    else:
        if w_0 is None:
            w_0 = np.zeros(p)

        optim_w, MSEs = gradient_descent(X, y, w_0, alpha_0, b, eps)

        return (optim_w, MSEs)