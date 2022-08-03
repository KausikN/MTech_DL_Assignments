'''
Eval Functions
'''

# Imports
import numpy as np

# Main Functions
# Eval Functions
# @ N Kausik CS21M037
def EvalFunc_Accuracy(y, t, **params):
    '''
    Calculates the accuracy of the model
    '''
    return np.sum(t == y)# / float(t.shape[0])