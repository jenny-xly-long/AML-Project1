import numpy as np

def gradient_descent(X, y, w_0, alpha_0, b, eps):

    # local variables
    i = 0 # iterations performed
    alpha = n_0 # step size
    prev_w = w_0 # weight of previous iteration
    current_w = w_0 # weight of current iteration

    # precomputes terms used in the gradient descent update
    X_T = X.T
    crossprod_X = X_T @ X
    y_term = X_T @ y

    # performs gradient descent until stopping condition reached
    while True:

        # updates the step size
        alpha = n_0/(1 + b*i)

        # updates the weights
        prev_w = current_w
        current_w = current_w - 2 * alpha * (crossprod_X @ current_w - y_term)

        # updates the iteration number
        i = i + 1

        # checks the stopping condition
        if np.norm(current_w - prev_w) < eps:
            break

    # returns the optimal weights
    return current_w
