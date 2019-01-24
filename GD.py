import numpy as np

def MSE(X, y, w):
    return np.mean((X @ w-y)**2)

def gradient_descent(X, y, w_0, alpha_0, b, eps):

    # local variables
    i = 0 # iterations performed
    alpha = alpha_0 # step size
    prev_w = w_0 # weight of previous iteration
    current_w = w_0 # weight of current iteration
    MSEs = list()


    # precomputes terms used in the gradient descent update
    X_T = X.T
    crossprod_X = X_T @ X
    y_term = X_T @ y

    # performs gradient descent until stopping condition reached
    while True:

        # updates the step size
        alpha = alpha_0/(1 + b*i)

        # updates the weights
        prev_w = current_w
        current_w = current_w - 2 * alpha * (crossprod_X @ current_w - y_term)

        # updates MSEs
        if i % 10  == 0:
            MSEs.append(MSE(X, y, current_w))

        # updates the iteration number
        i = i + 1

        # checks the stopping condition
        if np.linalg.norm(current_w - prev_w) < eps:
            break

    # returns the optimal weights
    return (current_w, MSEs)