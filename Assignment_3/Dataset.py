'''
Dataset
'''

# Imports
import os
import numpy as np
np.random.seed(0)
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Main Vars
DATASET_PATH_DAKSHINA = "Dataset/dakshina_dataset_v1.0"
DATASET_DAKSHINA_LANGS = [
    "bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"
]
# DATASET_PATH_DAKSHINA_TAMIL = "Dataset/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.{}.tsv"
DATASET_PATH_DAKSHINA_TAMIL = "Dataset/dakshina_dataset_v1.0/"
DATASET_PATH_DAKSHINA_TAMIL_PROCESSED = "Dataset/dakshina_processed/ta/"
DATASET_DAKSHINA_TAMIL_CLASSES = ["target", "input", "rel"]
DATASET_DAKSHINA_TAMIL_MAX_CHARS = {
    "input": 32,
    "target": 32, 
}
DATASET_DAKSHINA_TAMIL_UNIQUE_CHARS = {
    "input": 28,
    "target": 48,
}
SYMBOLS = {
    "start": "\t",
    "end": "\n"
}
FONT_PATH_TAMIL = "Library/Latha.ttf"

LOAD_SAVED_DATASET = True

# Main Functions
# Encoding Functions
def AddStartEndTokens(s):
    '''
    Add the Start and End Tokens to the String
    '''
    return SYMBOLS["start"] + s + SYMBOLS["end"]

def EncodeDataset_CharOneHot(dataset, input_MAX_CHARS=None, target_MAX_CHARS=None):
    '''
    Encode Dataset using One Hot per Character
    '''
    # Remove nan rows
    dataset = dataset.dropna().reset_index(drop=True)
    # Add SOS and EOS
    for c in ["input", "target"]:
        dataset[c] = dataset[c].apply(AddStartEndTokens)
    # Get Unique Charecters
    input_fullStr = str(dataset.input.str.cat(sep=""))
    input_chars = sorted(list(set(input_fullStr)))
    target_fullStr = str(dataset.target.str.cat(sep=""))
    target_chars = sorted(list(set(target_fullStr)))
    # Get Values
    input_MAX_CHARS = max([len(x) for x in dataset.input]) if input_MAX_CHARS is None else input_MAX_CHARS
    target_MAX_CHARS = max([len(x) for x in dataset.target]) if target_MAX_CHARS is None else target_MAX_CHARS
    input_N_CHARS = len(input_chars)
    target_N_CHARS = len(target_chars)
    print("Input Max Chars:", input_MAX_CHARS)
    print("Target Max Chars:", target_MAX_CHARS)
    print("Input N Chars:", input_N_CHARS, ":\n", input_chars)
    print("Target N Chars:", target_N_CHARS, ":\n", target_chars)
    # Construct Char Map
    input_char_map = {c: i+1 for i, c in enumerate(input_chars)}
    target_char_map = {c: i+1 for i, c in enumerate(target_chars)}
    input_chars = {i+1: c for i, c in enumerate(input_chars)}
    target_chars = {i+1: c for i, c in enumerate(target_chars)}
    input_chars[0], target_chars[0] = "", ""
    # Get One Hot Encoding
    dataset_size = dataset.shape[0]
    encoder_input_oneHot = np.zeros((dataset_size, input_MAX_CHARS, input_N_CHARS+1))
    decoder_input_oneHot = np.zeros((dataset_size, target_MAX_CHARS, target_N_CHARS+1))
    decoder_output_oneHot = np.zeros((dataset_size, target_MAX_CHARS, target_N_CHARS+1))
    for i in range(dataset_size):
        for j in range(len(dataset["input"][i])):
            encoder_input_oneHot[i, j, input_char_map[dataset["input"][i][j]]] = 1.0
        for j in range(len(dataset["target"][i])):
            decoder_input_oneHot[i, j, target_char_map[dataset["target"][i][j]]] = 1.0
            if j > 0: decoder_output_oneHot[i, j - 1, target_char_map[dataset["target"][i][j]]] = 1.0
    print("Encoder Input One Hot Shape:", encoder_input_oneHot.shape)
    print("Decoder Input One Hot Shape:", decoder_input_oneHot.shape)
    print("Decoder Output One Hot Shape:", decoder_output_oneHot.shape)
    
    dataset_onehot = {
        "encoder_input": encoder_input_oneHot,
        "decoder_input": decoder_input_oneHot,
        "decoder_output": decoder_output_oneHot,
        "chars": {
            "input_chars": input_chars,
            "target_chars": target_chars,
            "input_char_map": input_char_map,
            "target_char_map": target_char_map,
        }
    }
    return dataset_onehot

# Load Train and Test Dataset Functions
def LoadTrainDataset_Dakshina(
    path, 
    save_dir=DATASET_PATH_DAKSHINA_TAMIL_PROCESSED, **params
    ):
    '''
    Load Train Dakshina Dataset
    '''
    path_dataset = os.path.join(save_dir, "dataset.pkl")
    path_dataset_encoded = os.path.join(save_dir, "dataset_encoded.pkl")

    # Load if Available
    if LOAD_SAVED_DATASET:
        if os.path.exists(path_dataset) and os.path.exists(path_dataset_encoded):
            dataset = pickle.load(open(path_dataset, "rb"))
            dataset_encoded = pickle.load(open(path_dataset_encoded, "rb"))
            return dataset, dataset_encoded

    # Load Train Dataset
    train_dataset = pd.read_csv(path.format("train"), sep="\t", header=None, names=DATASET_DAKSHINA_TAMIL_CLASSES)
    # Get One Hot Encoding
    train_dataset_onehot = EncodeDataset_CharOneHot(
        train_dataset,
        input_MAX_CHARS=DATASET_DAKSHINA_TAMIL_MAX_CHARS["input"],
        target_MAX_CHARS=DATASET_DAKSHINA_TAMIL_MAX_CHARS["target"]
    )
    
    # Load Val Dataset
    val_dataset = pd.read_csv(path.format("dev"), sep="\t", header=None, names=DATASET_DAKSHINA_TAMIL_CLASSES)
    # Get One Hot Encoding
    val_dataset_onehot = EncodeDataset_CharOneHot(
        val_dataset,
        input_MAX_CHARS=DATASET_DAKSHINA_TAMIL_MAX_CHARS["input"],
        target_MAX_CHARS=DATASET_DAKSHINA_TAMIL_MAX_CHARS["target"]
    )

    dataset = {
        "train": train_dataset,
        "val": val_dataset
    }
    dataset_encoded = {
        "train": train_dataset_onehot,
        "val": val_dataset_onehot
    }
    # Save
    pickle.dump(dataset, open(path_dataset, "wb"))
    pickle.dump(dataset_encoded, open(path_dataset_encoded, "wb"))

    return dataset, dataset_encoded

def LoadTestDataset_Dakshina(
    path, 
    save_dir=DATASET_PATH_DAKSHINA_TAMIL_PROCESSED, **params
    ):
    '''
    Load Test Dakshina Dataset
    '''
    path_dataset = os.path.join(save_dir, "test_dataset.pkl")
    path_dataset_encoded = os.path.join(save_dir, "test_dataset_encoded.pkl")

    # Load if Available
    if LOAD_SAVED_DATASET:
        if os.path.exists(path_dataset) and os.path.exists(path_dataset_encoded):
            test_dataset = pickle.load(open(path_dataset, "rb"))
            test_dataset_onehot = pickle.load(open(path_dataset_encoded, "rb"))
            return test_dataset, test_dataset_onehot

    # Load Dataset
    test_dataset = pd.read_csv(path.format("test"), sep="\t", header=None, names=DATASET_DAKSHINA_TAMIL_CLASSES)
    # Get One Hot Encoding
    test_dataset_onehot = EncodeDataset_CharOneHot(
        test_dataset,
        input_MAX_CHARS=DATASET_DAKSHINA_TAMIL_MAX_CHARS["input"],
        target_MAX_CHARS=DATASET_DAKSHINA_TAMIL_MAX_CHARS["target"]
    )
    # Save
    pickle.dump(test_dataset, open(path_dataset, "wb"))
    pickle.dump(test_dataset_onehot, open(path_dataset_encoded, "wb"))
    
    return test_dataset, test_dataset_onehot

# Run