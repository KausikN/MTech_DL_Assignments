'''
Loss Functions
'''

# Imports
import numpy as np

# Main Functions
# Loss Functions
# @ Karthikeyan S CS21M028

# Derivatives Functions
def LossFunc_MeanSquaredError(y, t, **params):
    '''
    Mean Squared Error
    '''
    loss = 0.5 * np.sum((y-t)**2, axis=-1)
    loss = np.mean(loss)
    return loss

def LossFunc_CrossEntropy(y, t, **params):
    '''
    Cross Entropy
    '''
    y = np.clip(y, 1e-15, 1.0)
    loss = -np.sum(t * np.log(y + (t == 0)), axis=-1)
    loss = np.mean(loss)
    return loss

# Derivatives Functions
def LossFuncDeriv_MeanSquaredError_WithSoftmax(y, t, **params):
    '''
    Mean Squared Error Derivative
    '''
    return (y - t) * y * (1 - y)

def LossFuncDeriv_CrossEntropy_WithSoftmax(y, t, **params):
    '''
    Cross Entropy Derivative
    '''
    return y - t

# Main Vars
LOSS_FUNCTIONS = {
    "mse": {
        "func": LossFunc_MeanSquaredError,
        "deriv": LossFuncDeriv_MeanSquaredError_WithSoftmax
    },
    "cross_entropy": {
        "func": LossFunc_CrossEntropy,
        "deriv": LossFuncDeriv_CrossEntropy_WithSoftmax
    }
}