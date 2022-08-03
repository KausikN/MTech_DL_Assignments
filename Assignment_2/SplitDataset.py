'''
Split Dataset into Train and validation
'''

# Imports
import argparse
from Dataset import *


# Main Functions
def Runner_ParseArgs():
    '''
    Parse Args
    '''
    global DATASET_PATH_INATURALIST
    
    parser = argparse.ArgumentParser(description="Training and Testing for DL Assignment 2 Part B")
    parser.add_argument("--dataset", "-dt", type=str, default=DATASET_PATH_INATURALIST, help="Dataset path to use")
    args = parser.parse_args()
    DATASET_PATH_INATURALIST = args.dataset
    return args

# Run
if __name__ == "__main__":
    # Parse Args
    ARGS = Runner_ParseArgs()
    # Run
    CreateValidationDataset_INaturalist(DATASET_PATH_INATURALIST, DATASET_INATURALIST_CLASSES, validation_split=0.1)