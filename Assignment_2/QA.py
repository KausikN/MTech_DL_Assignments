'''
Questions Part A
'''

# Imports
import json
import argparse
import functools
from pprint import pprint

from Model import *
from TestModel import *
from GuidedBackprop import *
from VisualiseFilters import *

# Main Functions
# Wandb Sweep Function for Part A @ Karthikeyan S CS21M028
def Model_Sweep_Run(wandb_data):
    '''
    Part A Model Sweep Runner
    '''
    # Init
    wandb.init()

    # Get Run Config
    config = wandb.config
    N_EPOCHS = config.n_epochs
    BATCH_SIZE = config.batch_size

    N_FILTERS = config.n_filters
    FILTER_SIZE = config.filter_size
    DROPOUT = config.dropout
    BATCH_NORM = config.batch_norm
    DENSE_NEURONS = config.dense_neurons
    DENSE_DROPOUT = config.dense_dropout

    LEARNING_RATE = config.lr

    print("RUN CONFIG:")
    pprint(config)

    # Get Inputs
    N_FILTERS_LAYERWISE = [int(N_FILTERS[i]) for i in range(5)]
    inputs = {
        "img_size": (227, 227, 3), 
        "Y_shape": len(DATASET_INATURALIST_CLASSES), 
        "model": {
            "blocks": [
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[0], conv_kernel_size=(FILTER_SIZE[0], FILTER_SIZE[0]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[1], conv_kernel_size=(FILTER_SIZE[1], FILTER_SIZE[1]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[2], conv_kernel_size=(FILTER_SIZE[2], FILTER_SIZE[2]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[3], conv_kernel_size=(FILTER_SIZE[3], FILTER_SIZE[3]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
                functools.partial(Block_CRM, conv_filters=N_FILTERS_LAYERWISE[4], conv_kernel_size=(FILTER_SIZE[4], FILTER_SIZE[4]), batch_norm=BATCH_NORM, 
                    act_fn="relu", maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), dropout_rate=DROPOUT), 
            ], 
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
        shuffle=True, data_aug=True
    )
    inputs["dataset"] = DATASET

    # Build Model
    MODEL = Model_SequentialBlocks(
        X_shape=inputs["img_size"], Y_shape=inputs["Y_shape"], 
        Blocks=inputs["model"]["blocks"],
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
    
    parser = argparse.ArgumentParser(description="Training and Testing for DL Assignment 2 Part A")

    parser.add_argument("--mode", "-m", type=str, default="train", help="train | test | gb | vf")
    parser.add_argument("--model", "-ml", type=str, default="Models/Model_PartA.h5", help="Model path to use or save to")
    parser.add_argument("--dataset", "-dt", type=str, default=DATASET_PATH_INATURALIST, help="Dataset path to use")

    # Train Args
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")

    parser.add_argument("--filter_size", "-fs", type=str, default="3,3,3,5,7", help="Filter sizes")
    parser.add_argument("--n_filters", "-nf", type=str, default="32,64,64,128,128", help="Number of filters")
    parser.add_argument("--dropout", "-dp", type=float, default=0.1, help="Layerwise dropout")
    parser.add_argument("--no_batch_norm", "-nbn", action='store_false', help="Dont use Batch Normalisation")
    parser.add_argument("--dense_neurons", "-dn", type=int, default=512, help="Number of Dense Layer Neurons")
    parser.add_argument("--dense_dropout", "-dd", type=float, default=0.1, help="Dense Layer Dropout")

    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate")

    # Guided Backprop and Visualise Filters Args
    parser.add_argument("--n_cols", "-nc", type=int, default=4, help="Number of columns in display")
    parser.add_argument("--filter_rgb", "-frgb", action='store_true', help="Display Filters as RGB image")

    args = parser.parse_args()
    DATASET_PATH_INATURALIST = args.dataset
    return args

def Runner_PartA_Train(args):
    '''
    Train Model
    '''
    # Load Wandb Data
    WANDB_DATA = json.load(open("config.json", "r"))
    # Sweep Setup
    SWEEP_CONFIG = {
        "name": "part-a-run-1",
        "method": "grid",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "n_epochs": {
                "values": [args.epochs]
            },
            "batch_size": {
                "values": [args.batch_size]
            },

            "filter_size": {
                "values": [
                    [int(x) for x in args.filter_size.split(",")]
                ]
            },
            "n_filters": {
                "values": [
                    [int(x) for x in args.n_filters.split(",")]
                ]
            },
            "dropout": {
                "values": [args.dropout]
            },
            "batch_norm": {
                "values": [args.no_batch_norm]
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

def Runner_PartA_Test(args):
    '''
    Test Model
    '''
    TestModel_RunTest(Model_LoadModel(args.model), X_shape=(227, 227, 3), Y_shape=len(DATASET_INATURALIST_CLASSES))

def Runner_PartA_GuidedBackprop(args):
    '''
    Guided Backprop on model
    '''
    GuidedBackprop_Display(Model_LoadModel(args.model), X_shape=(227, 227, 3), Y_shape=len(DATASET_INATURALIST_CLASSES), nCols=3, dataset_path=DATASET_PATH_INATURALIST)

def Runner_PartA_VisualiseFilters(args):
    '''
    Visualise Filters of model
    '''
    VisualiseFilter_Display(Model_LoadModel(args.model), rgb=args.filter_rgb, nCols=args.n_cols)

# Run
if __name__ == "__main__":
    # Parse Args
    ARGS = Runner_ParseArgs()
    # Run
    if ARGS.mode == "train":
        Runner_PartA_Train(ARGS)
    elif ARGS.mode == "test":
        Runner_PartA_Test(ARGS)
    elif ARGS.mode == "gb":
        Runner_PartA_GuidedBackprop(ARGS)
    elif ARGS.mode == "vf":
        Runner_PartA_VisualiseFilters(ARGS)
    else:
        print("Invalid Mode!")