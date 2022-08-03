'''
Activation Functions
'''

# Imports
import numpy as np
from scipy.special import expit

# Main Functions
# Activation Functions
# @ N Kausik CS21M037
def ActFunc_sigmoid(x):
    '''
    Sigmoid Function
    '''
    # return 1 / (1 + np.exp(-x))
    return expit(x)


def ActFunc_tanh(x):
    '''
    Tanh Function
    '''
    return np.tanh(x)

def ActFunc_relu(x):
    '''
    ReLU Function
    '''
    return np.maximum(0, x, x)

def ActFunc_softmax(x):
    '''
    Softmax Function
    '''
    s = np.zeros(x.shape)
    for i in range(x.shape[1]):
        x_e = np.exp(x[:, i] - np.max(x[:, i]))
        s[:, i] = x_e / np.sum(x_e)
    return s


# Derivatives Functions
# @ N Kausik CS21M037
def ActFuncDeriv_sigmoid(x):
    '''
    Sigmoid Derivative Function
    '''
    x_sigma = ActFunc_sigmoid(x)
    return x_sigma * (1 - x_sigma)

def ActFuncDeriv_tanh(x):
    '''
    Tanh Derivative Function
    '''
    x_tanh = ActFunc_tanh(x)
    return 1 - np.square(x_tanh)

def ActFuncDeriv_relu(x):
    '''
    ReLU Derivative Function
    '''
    x[x <= 0] = 0.0
    x[x > 0] = 1.0
    return x

def ActFuncDeriv_softmax(x):
    '''
    Softmax Derivative Function
    '''
    return np.ones(x.shape)

# Main Vars
ACTIVATION_FUNCTIONS = {
    "sigmoid": {
        "func": ActFunc_sigmoid,
        "deriv": ActFuncDeriv_sigmoid
    },
    "tanh": {
        "func": ActFunc_tanh,
        "deriv": ActFuncDeriv_tanh
    },
    "relu": {
        "func": ActFunc_relu,
        "deriv": ActFuncDeriv_relu
    },
    "softmax": {
        "func": ActFunc_softmax,
        "deriv": ActFuncDeriv_softmax
    }
}