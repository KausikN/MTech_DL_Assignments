'''
Questions Part B
'''

# Imports
import json
import argparse
import functools
from pprint import pprint

from Model import *
from TestModel import *

# Main Functions
# Wandb Sweep Function for Part B @ N Kausik CS21M037
def Model_Sweep_Run(wandb_data):
    '''
    Part B Model Sweep Runner
    '''
    # Init
    wandb.init()

    # Get Run Config
    config = wandb.config
    MODEL_NAME = config.model_name
    N_EPOCHS = config.n_epochs
    BATCH_SIZE = config.batch_size
    DATA_AUG = config.data_aug
    UNFREEZE_COUNT = config.unfreeze_count

    DENSE_NEURONS = config.dense_neurons
    DENSE_DROPOUT = config.dense_dropout

    LEARNING_RATE = config.lr

    print("RUN CONFIG:")
    pprint(config)

    # Get Inputs
    inputs = {
        "img_size": (224, 224, 3), 
        "Y_shape": len(DATASET_INATURALIST_CLASSES), 
        "model": {
            "compile_params": {
                "loss_fn": "categorical_crossentropy",
                "optimizer": Adam(learning_rate=LEARNING_RATE),
                "metrics": ["accuracy"]
            }
        }
    }

    # Get Train Val Dataset
    DATASET = LoadTrainDataset_INaturalist(
        DATASET_PATH_INATURALIST,  
        img_size=inputs["img_size"][:2], batch_size=BATCH_SIZE, 
        shuffle=True, data_aug=DATA_AUG
    )
    inputs["dataset"] = DATASET

    # Build Model
    MODEL = Model_PretrainedBlocks(
        X_shape=inputs["img_size"], Y_shape=inputs["Y_shape"], 
        model_name=MODEL_NAME,
        unfreeze_count=UNFREEZE_COUNT,
        dense_n_neurons=DENSE_NEURONS, dense_dropout_rate=DENSE_DROPOUT
    )
    MODEL = Model_Compile(MODEL, **inputs["model"]["compile_params"])

    # Train Model
    TRAINED_MODEL, TRAIN_HISTORY = Model_Train(MODEL, inputs, N_EPOCHS, wandb_data)

    # Load Best Model
    TRAINED_MODEL = Model_LoadModel("Models/best_model.h5")
    # Get Test Dataset
    DATASET_PATH_INATURALIST_TEST = os.path.join(DATASET_PATH_INATURALIST, "val")
    DATASET_TEST = LoadTestDataset_INaturalist(
        DATASET_PATH_INATURALIST_TEST, 
        img_size=inputs["img_size"][:2], batch_size=BATCH_SIZE, 
        shuffle=False
    )
    # Test Best Model
    loss_test, eval_test = Model_Test(TRAINED_MODEL, DATASET_TEST)

    # Wandb log test data
    wandb.log({
        "loss_test": loss_test,
        "eval_test": eval_test
    })

    # Close Wandb Run
    # run_name = "ep:"+str(N_EPOCHS) + "_" + "bs:"+str(BATCH_SIZE) + "_" + "nf:"+str(N_FILTERS) + "_" + str(DROPOUT)
    # wandb.run.name = run_name
    wandb.finish()

# Runner Functions
def Runner_ParseArgs():
    '''
    Parse Args
    '''
    global DATASET_PATH_INATURALIST
    
    parser = argparse.ArgumentParser(description="Training and Testing for DL Assignment 2 Part B")

    parser.add_argument("--mode", "-m", type=str, default="train", help="train or test")
    parser.add_argument("--model", "-ml", type=str, default="Models/Model_PartB.h5", help="Model path to use or save to")
    parser.add_argument("--dataset", "-dt", type=str, default=DATASET_PATH_INATURALIST, help="Dataset path to use")

    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Batch size")
    parser.add_argument("--no_data_aug", "-nda", action='store_false', help="Dont use Data Augmentation")

    parser.add_argument("--unfreeze_count", "-uc", type=int, default=10, help="Number of top layers to unfreeze")
    parser.add_argument("--dense_neurons", "-dn", type=int, default=512, help="Number of Dense Layer Neurons")
    parser.add_argument("--dense_dropout", "-dd", type=float, default=0.1, help="Dense Layer Dropout")

    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0001, help="Learning rate")

    args = parser.parse_args()
    DATASET_PATH_INATURALIST = args.dataset
    return args

def Runner_PartB_Train(args):
    '''
    Pretrain/Finetune Model
    '''
    # Load Wandb Data
    WANDB_DATA = json.load(open("config.json", "r"))
    # Sweep Setup
    SWEEP_CONFIG = {
        "name": "part-b-run-1",
        "method": "grid",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "model_name":{
                "values": ["Xception"]
            },

            "n_epochs": {
                "values": [args.epochs]
            },
            "batch_size": {
                "values": [args.batch_size]
            },
            "data_aug":{
                "values": [args.no_data_aug]
            },

            "unfreeze_count":{
                "values": [args.unfreeze_count]
            },

            "dense_neurons": {
                "values": [args.dense_neurons]
            },
            "dense_dropout": {
                "values": [args.dense_dropout]
            },

            "lr": {
                "values": [args.learning_rate]
            }
        }
    }
    # Run Sweep
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"])
    # sweep_id = ""
    TRAINER_FUNC = functools.partial(Model_Sweep_Run, wandb_data=WANDB_DATA)
    wandb.agent(sweep_id, TRAINER_FUNC, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"], count=1)
    # Save Model
    Model_SaveModel(Model_LoadModel("Models/best_model.h5"), args.model)

def Runner_PartB_Test(args):
    '''
    Test Model
    '''
    TestModel_RunTest(Model_LoadModel(args.model), X_shape=(224, 224, 3), Y_shape=len(DATASET_INATURALIST_CLASSES))

# Run
if __name__ == "__main__":
    # Parse Args
    ARGS = Runner_ParseArgs()
    # Run
    if ARGS.mode == "train":
        Runner_PartB_Train(ARGS)
    elif ARGS.mode == "test":
        Runner_PartB_Test(ARGS)
    else:
        print("Invalid Mode!")