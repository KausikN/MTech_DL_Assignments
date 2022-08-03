'''
Init Functions
'''

# Imports
import numpy as np
np.random.seed(0)

# Main Functions
# Init Functions
# @ Karthikeyan S CS21M028
# Done: Zeros, Random, Xavier
def InitFunc_Zeros(layer_sizes, **params):
    '''
    Intializes the weights and biases with zeros
    '''
    Ws = []
    bs = []
    for i in range(len(layer_sizes)-1):
        W = np.zeros((layer_sizes[i], layer_sizes[i+1]))
        b = np.zeros((1, layer_sizes[i+1]))
        Ws.append(W)
        bs.append(b)
    return Ws, bs

def InitFunc_Random(layer_sizes, **params):
    '''
    Intializes the weights and biases with random values
    '''
    Ws = []
    bs = []
    randomRange = params["randomRange"] if "randomRange" in params else [0.0, 1.0]
    for i in range(len(layer_sizes)-1):
        W = np.random.uniform(randomRange[0], randomRange[1], (layer_sizes[i], layer_sizes[i+1]))
        b = np.random.uniform(randomRange[0], randomRange[1], (1, layer_sizes[i+1]))
        Ws.append(W)
        bs.append(b)
    return Ws, bs

def InitFunc_Xavier(layer_sizes, **params):
    '''
    Intializes the weights and biases with Xavier initialization
    '''
    Ws = []
    bs = []
    for i in range(len(layer_sizes)-1):
        var_xavier = 2.0 / np.sqrt(layer_sizes[i] + layer_sizes[i+1])
        W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * var_xavier
        b = np.zeros((layer_sizes[i+1], 1))
        Ws.append(W)
        bs.append(b)
    return Ws, bs

# Main Vars
INIT_FUNCTIONS = {
    "zeros": InitFunc_Zeros,
    "random": InitFunc_Random,
    "xavier": InitFunc_Xavier
}