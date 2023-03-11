import numpy as np
import math


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    g = 1/(1 + np.exp(-z))
    return g


def compute_cost_reg(X, y, w, b, lambda_, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    z = np.dot(X, w) + b

    # Prevent zeros and infinitis
    f_wb = sigmoid(z)
    epsilon = 1e-8
    f_wb = np.clip(f_wb, epsilon, 1-epsilon)

    total_cost = np.sum(-y * np.log(f_wb) - (1 - y) * np.log(1-f_wb))/m
    total_cost += np.sum(np.power(w, 2))*(lambda_/(2*m))

    return total_cost


def compute_gradient_reg(X, y, w, b, lambda_, *argv):
    """
    Computes the gradient for logistic regression 

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
      *argv : unused, for compatibility with regularized version 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    err = f_wb - y

    for i in range(n):
        dj_dw[i] = np.sum(err * X[:, i])/m

    dj_db = np.sum(err)/m
    dj_dw += (w*(lambda_ / m))

    return dj_db, dj_dw


def gradient_descent_reg(X, y, w_in, b_in, alpha, num_iters, lambda_):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent

    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(X)

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_reg(X, y, w_in, b_in, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            cost = compute_cost_reg(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    # return w and J,w history for graphing
    return w_in, b_in, J_history, w_history


def predict(X, w, b):
    '''
    Calculate the probability of the answer be 1

    Args:
        X : (ndarray Shape (m, n) data, m examples by n features
        w : (ndarray Shape (n,))  Weights of parameters of the model
        b : (scalar)              Bias of the model
    Output:
        f_wb: (ndarray Shape (m,)) binary prediction
        g: (ndarray Shape (m,)) probability prediction of the values
    '''

    g = np.dot(X, w) + b
    f_wb = np.where(g >= 0.5, 1.0, 0.0)
    return f_wb, g
