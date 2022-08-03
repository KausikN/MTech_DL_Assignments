'''
Dataset Functions
'''

# Imports
import math
import numpy as np
from tqdm import tqdm
from keras.datasets import fashion_mnist, mnist

# Main Vars
DATASET_N_CLASSES = 10
DATASET_FASHION_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
DATASET_MNIST_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Main Functions
# Encoding Functions
def OneHotEncoder(y, num_classes=10):
    y = np.array(y)
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot

# Dataset Load Functions
def LoadFashionDataset():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = np.array(X_train) / 255.0
    X_test = np.array(X_test) / 255.0
    return X_train, X_test, Y_train, Y_test

def LoadMNISTDataset():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.array(X_train) / 255.0
    X_test = np.array(X_test) / 255.0
    return X_train, X_test, Y_train, Y_test

# Dataset Split Functions
def SplitDataset_ClassWise(X, Y):
    X_classwise = {}
    for x, y in tqdm(zip(X, Y)):
        if y not in X_classwise.keys():
            X_classwise[y] = []
        X_classwise[y].append(x)
    return X_classwise

def SplitDataset_TrainValClasswise(X_classwise, train_size=0.9):
    X_train = []
    X_val = []
    Y_train = []
    Y_val = []
    
    for i in tqdm(X_classwise.keys()):
        split_index = math.ceil(len(X_classwise[i]) * train_size)
        X_train.extend(X_classwise[i][:split_index])
        X_val.extend(X_classwise[i][split_index:])
        Y_train.extend([i] * split_index)
        Y_val.extend([i] * (len(X_classwise[i]) - split_index))

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    Y_train = np.array(Y_train)
    Y_val = np.array(Y_val)
    
    return X_train, Y_train, X_val, Y_val

def FlattenData(X):
    X = np.array(X)
    X = X.reshape(X.shape[0], -1)
    return X

# Normalization Functions
def NormalizeData(X, normRange=(-1.0, 1.0)):
    X = np.array(X)
    X_min = np.min(X)
    X_max = np.max(X)
    X = (X - X_min) / (X_max - X_min)
    X = X * (normRange[1] - normRange[0]) + normRange[0]
    return X