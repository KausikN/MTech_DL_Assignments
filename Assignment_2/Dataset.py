'''
Dataset
'''

# Imports
import os
import shutil
import numpy as np
np.random.seed(0)
from tqdm import tqdm
from keras.preprocessing import *
from keras.preprocessing.image import *


# Main Vars
DATASET_PATH_INATURALIST = "Dataset/inaturalist_12K"
DATASET_INATURALIST_CLASSES = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]

# Main Functions
# Load Train and Test Dataset Functions @ Karthikeyan S CS21M028
def LoadTrainDataset_INaturalist(
    path, 
    img_size=(227, 227), batch_size=128, 
    shuffle=True, data_aug=True
    ):
    '''
    Load Train INaturalist Dataset
    '''
    # Load Train Dataset
    if data_aug:
        dataset_train = ImageDataGenerator(
            rescale=1.0/255,
            # Data Augmentation
            height_shift_range=0.2,
            width_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode="nearest"
        )
    else:
        dataset_train = ImageDataGenerator(rescale=1.0/255)
    # Get Train Generator
    train_generator = dataset_train.flow_from_directory(
        os.path.join(path, "train"), 
        target_size=img_size, class_mode='categorical', 
        batch_size=batch_size, 
        shuffle=shuffle, seed=42
    )
    # Load Val Dataset
    dataset_val = ImageDataGenerator(
        rescale=1.0/255
    )
    # Get Val Generator
    val_generator = dataset_val.flow_from_directory(
        os.path.join(path, "validation"), 
        target_size=img_size, class_mode='categorical', 
        batch_size=batch_size, 
        shuffle=shuffle, seed=42
    )
    dataset = {
        "train": train_generator,
        "val": val_generator
    }

    return dataset

def LoadTestDataset_INaturalist(
    path, 
    img_size=(227, 227),
    batch_size=128, 
    shuffle=True
    ):
    '''
    Load Test INaturalist Dataset
    '''
    # Load Dataset
    dataset = ImageDataGenerator(
        rescale=1.0/255
    )
    # Get Test Generator
    dataset_test = dataset.flow_from_directory(
        path, 
        target_size=img_size, class_mode='categorical', 
        batch_size=batch_size, 
        shuffle=shuffle, seed=42
    )
    
    return dataset_test

# Split Train into Train and Val Dataset @ N Kausik CS21M037
def CreateValidationDataset_INaturalist(path, classes, validation_split=0.1):
    '''
    Create Validation Dataset
    '''
    train_path = os.path.join(path, "train")
    save_path = os.path.join(path, "validation")
    try:
        shutil.rmtree(save_path)
    except:
        pass
    os.mkdir(save_path)
    # Add Images
    for c in classes:
        os.mkdir(os.path.join(save_path, c))
        Is_train = os.listdir(os.path.join(train_path, c))
        Is_train_filtered = list(filter(lambda n: n != ".DS_Store", Is_train))
        count = len(Is_train_filtered)
        np.random.shuffle(Is_train_filtered)
        Is_val = Is_train_filtered[:round(validation_split*count)]
        for I_path in Is_val:
            shutil.move(os.path.join(train_path, c, I_path), os.path.join(save_path, c, I_path))

# Get Random Image Path from Dataset @ N Kausik CS21M037
def GetImagePath_Random(dataset_path):
    '''
    Get Random Image Path in train dataset
    '''
    dataset_path = os.path.join(dataset_path, "train")
    class_random = os.listdir(dataset_path)[np.random.randint(0, 10)]
    class_Is = os.listdir(os.path.join(dataset_path, class_random))
    I_name = class_Is[np.random.randint(0, len(class_Is))]
    I_path = os.path.join(dataset_path, class_random, I_name)
    return I_path, class_random

def GetTestImagePath_Random(c):
    '''
    Get Random Test Image Path in class c
    '''
    dataset_path = os.path.join(DATASET_PATH_INATURALIST, "val")
    class_Is = os.listdir(os.path.join(dataset_path, c))
    I_name = class_Is[np.random.randint(0, len(class_Is))]
    I_path = os.path.join(dataset_path, c, I_name)
    return I_path

# Run