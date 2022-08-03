'''
Guided Backpropogation
'''

# Imports
from Model import *

# Main Functions
# Test Model and print test loss and accuracy on a input model @ Karthikeyan S CS21M028
def TestModel_RunTest(MODEL, X_shape=(227, 227, 3), Y_shape=10):
    '''
    Test the model and display loss and accuracy
    '''
    DATASET_PATH_INATURALIST_TEST = os.path.join(DATASET_PATH_INATURALIST, "val")
    # Load Test Dataset
    DATASET_TEST = LoadTestDataset_INaturalist(
        DATASET_PATH_INATURALIST_TEST,  
        img_size=tuple(X_shape[:2]), batch_size=128, 
        shuffle=True
    )
    # Test Model
    loss_test, eval_test = Model_Test(MODEL, DATASET_TEST)
    print("MODEL TEST:")
    print("Loss:", loss_test)
    print("Accuracy:", eval_test)

# Run
# Params

# Params

# Run