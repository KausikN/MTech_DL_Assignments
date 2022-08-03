"""
Main Runner file for training and testing
"""

# Imports
import os
import json
from pprint import pprint
import argparse

from Dataset import *
from FNN import *

# Main Functions
# Arguments Parsing Functions
def Runner_ParseArgs():
    parser = argparse.ArgumentParser(description="Training and Testing for DL Assignment 1")

    parser.add_argument("--mode", "-m", type=str, default="train", help="train or test")
    parser.add_argument("--model", "-ml", type=str, default="models/model_1.p", help="Model path (.json) to use or save to")
    parser.add_argument("--dataset", "-d", type=str, default="fashion", help="Dataset to use (fashion or mnist)")

    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_layers", "-hl", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--hidden_neurons", "-hn", type=int, default=128, help="Number of hidden neurons per layer")

    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0005, help="Weight decay for L2 regularisation")

    parser.add_argument("--optimiser", "-o", type=str, default="nesterov", help="Optimiser")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", "-og", type=float, default=0.5, help="Gamma for optimiser")
    parser.add_argument("--eps", "-oeps", type=float, default=1e-8, help="Epsilon for optimiser")
    parser.add_argument("--beta1", "-ob1", type=float, default=0.9, help="Beta 1 for optimiser")
    parser.add_argument("--beta2", "-ob2", type=float, default=0.99, help="Beta 2 for optimiser")

    parser.add_argument("--init_func", "-if", type=str, default="xavier", help="Initialisation function")
    parser.add_argument("--act_func", "-af", type=str, default="tanh", help="Activation function")
    parser.add_argument("--loss_func", "-lf", type=str, default="cross_entropy", help="Loss function")

    parser.add_argument("--wandb", "-w", action='store_true', help="Use wandb")
    parser.add_argument("--verbose", "-v", action='store_true', help="Verbose")

    args = parser.parse_args()
    return args

# Runner Functions
def Runner_Train(args):
    # Get Wandb Data and Init
    WANDB_DATA = json.load(open("config.json", "r"))
    WANDB_DATA["use_wandb"] = args.wandb
    if WANDB_DATA["use_wandb"]:
        wandb.init(project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"])
    # Get Params
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    N_HIDDEN_LAYERS = args.hidden_layers
    N_HIDDEN_NEURONS = args.hidden_neurons

    WEIGHT_DECAY = args.weight_decay

    OPTIMISER_NAME = args.optimiser
    LEARNING_RATE = args.learning_rate
    GAMMA = args.gamma
    EPS = args.eps
    BETA1 = args.beta1
    BETA2 = args.beta2

    INIT_FUNC_NAME = args.init_func
    ACT_FUNC_NAME = args.act_func
    OUT_ACT_FUNC_NAME = "softmax"
    LOSS_FUNC_NAME = args.loss_func
    VERBOSE = args.verbose
    # Print
    if VERBOSE: pprint(args)
    
    # Get Dataset
    # Load Data
    LABELS = DATASET_FASHION_LABELS
    if args.dataset == "fashion":
        X_train_full, X_test, Y_train, Y_test = LoadFashionDataset()
        LABELS = DATASET_FASHION_LABELS
    elif args.dataset == "mnist":
        X_train_full, X_test, Y_train, Y_test = LoadMNISTDataset()
        LABELS = DATASET_MNIST_LABELS
    # Normalize Data
    X_train_full = NormalizeData(X_train_full, normRange=(0.0, 1.0))
    X_test = NormalizeData(X_test, normRange=(0.0, 1.0))
    # Split Data
    X_train_full_classwise = SplitDataset_ClassWise(X_train_full, Y_train)
    X_test_classwise = SplitDataset_ClassWise(X_test, Y_test)
    # Split into Train and Validation
    X_train, Y_train, X_val, Y_val = SplitDataset_TrainValClasswise(X_train_full_classwise, train_size=0.9)
    # Flatten Input and One Hot Output
    X_train_flat = FlattenData(X_train)
    X_val_flat = FlattenData(X_val)
    X_test_flat = FlattenData(X_test)
    Y_train_oh = OneHotEncoder(Y_train, DATASET_N_CLASSES)
    Y_val_oh = OneHotEncoder(Y_val, DATASET_N_CLASSES)
    Y_test_oh = OneHotEncoder(Y_test, DATASET_N_CLASSES)
    # Print
    if VERBOSE:
        print("Train:")
        print("X_train_full_classwise:", [len(X_train_full_classwise[i]) for i in range(DATASET_N_CLASSES)])
        print("X_train:", X_train.shape, X_train.min(), X_train.max())
        print("Y_train:", Y_train.shape)
        print("X_train_flat:", X_train_flat.shape, X_train_flat.min(), X_train_flat.max())
        print("Y_train_oh:", Y_train_oh.shape)
        print()
        print("Validation:")
        print("X_val:", X_val.shape, X_val.min(), X_val.max())
        print("Y_val:", Y_val.shape)
        print("X_val_flat:", X_train_flat.shape, X_train_flat.min(), X_train_flat.max())
        print("Y_train_oh:", Y_train_oh.shape)
        print()
        print("Test:")
        print("X_test_classwise:", [len(X_test_classwise[i]) for i in range(DATASET_N_CLASSES)])
        print("X_test:", X_test.shape, X_test.min(), X_test.max())
        print("Y_test:", Y_test.shape)
        print("X_test_flat:", X_test_flat.shape, X_test_flat.min(), X_test_flat.max())
        print("Y_test_oh:", Y_test_oh.shape)
        print()
    INPUTS = {
        "X": X_train_flat,
        "Y": Y_train_oh,
        "X_val": X_val_flat,
        "Y_val": Y_val_oh,
        "X_test": X_test_flat,
        "Y_test": Y_test_oh
    }

    HIDDEN_NETWORK_LAYERS = [N_HIDDEN_NEURONS] * N_HIDDEN_LAYERS
    NETWORK_LAYERS = [INPUTS["X"].shape[1]] + HIDDEN_NETWORK_LAYERS + [INPUTS["Y"].shape[1]]

    ACT_FUNC = ACTIVATION_FUNCTIONS[ACT_FUNC_NAME]
    OUT_ACT_FUNC = ACTIVATION_FUNCTIONS[OUT_ACT_FUNC_NAME]
    ACT_FUNC["params"] = {}
    OUT_ACT_FUNC["params"] = {}

    FUNCS = {
        # Initialization Functions
        "init_fn": {
            "func": INIT_FUNCTIONS[INIT_FUNC_NAME],
            "params": {
                "randomRange": [-1.0, 1.0]
            }
        },

        # Activation Functions
        "act_fns": 
            # Hidden Layer Activation Functions
            [ACT_FUNC] * len(HIDDEN_NETWORK_LAYERS) + 
            # Output Layer Activation Function
            [OUT_ACT_FUNC],

        # Loss Function
        "loss_fn": {
            "func": LOSS_FUNCTIONS[LOSS_FUNC_NAME]["func"],
            "deriv": LOSS_FUNCTIONS[LOSS_FUNC_NAME]["deriv"],
            "params": {
                "l1_lambda": 0.0,
                "l2_lambda": WEIGHT_DECAY
            }
        },

        # Update Function
        "update_fn": {
            "func": OPTIMISERS[OPTIMISER_NAME],
            "params": {
                "lr": LEARNING_RATE,
                "gamma": GAMMA,
                "eps": EPS,
                "beta1": BETA1,
                "beta2": BETA2
            },
            "data": {}
        },

        # Evaluation Function
        "eval_fn": {
            "func": EvalFunc_Accuracy,
            "params": {},
        }
    }
    if VERBOSE:
        print("FUNCS:")
        pprint(FUNCS)

    # Train Model
    TRAINED_PARAMETERS, TRAIN_HISTORY = Model_Train(INPUTS, NETWORK_LAYERS, FUNCS, N_EPOCHS, BATCH_SIZE, WANDB_DATA)

    # Test
    X_test = INPUTS["X_test"]
    Y_test = INPUTS["Y_test"]
    # Test Loss
    W_flat = np.concatenate([w.flatten() for w in TRAINED_PARAMETERS["Ws"]])
    b_flat = np.concatenate([b.flatten() for b in TRAINED_PARAMETERS["bs"]])
    params_flat = np.concatenate([W_flat, b_flat])
    loss_reg = TRAINED_PARAMETERS["loss_fn"]["params"]["l2_lambda"] * np.sqrt(np.sum(params_flat**2)) + \
                TRAINED_PARAMETERS["loss_fn"]["params"]["l1_lambda"] * np.sum(np.abs(params_flat))
    Y_test_out, As, Os = ForwardPropogation(X_test, TRAINED_PARAMETERS)
    loss_test = loss_reg + TRAINED_PARAMETERS["loss_fn"]["func"](Y_test_out.T, Y_test, **TRAINED_PARAMETERS["loss_fn"]["params"])
    print("Test Loss:", loss_test)
    # Test Eval
    Y_test_pred = np.argmax(Y_test_out.T, axis=-1)
    Y_test = np.argmax(Y_test, axis=-1)
    eval_test = FUNCS["eval_fn"]["func"](Y_test_pred, Y_test, **FUNCS["eval_fn"]["params"])
    print("Test Eval:", eval_test, "/", Y_test.shape[0])

    # Save Model
    if not (args.model == ""):
        Model_Save(TRAINED_PARAMETERS, args.model)
    # Wandb Log
    if WANDB_DATA["use_wandb"]:
        # Wandb log test data
        wandb.log({
            "loss_test": loss_test,
            "eval_test": eval_test
        })
        # Wandb log confusion matrix
        wandb.log({
            "confusion_matrix": Wandb_LogConfusionMatrix(Y_test_pred, Y_test, LABELS)
        })
        # Close Wandb Run
        wandb.finish()

def Runner_Test(args):
    # Get Params
    MODEL_PATH = args.model
    VERBOSE = args.verbose
    # Print
    if VERBOSE: pprint(args)
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print("Invalid Model Path.")
        return
    TRAINED_PARAMETERS = Model_Load(MODEL_PATH)
    # Get Dataset
    # Load Data
    if args.dataset == "fashion":
        X_train, X_test, Y_train, Y_test = LoadFashionDataset()
    elif args.dataset == "mnist":
        X_train, X_test, Y_train, Y_test = LoadMNISTDataset()
    # Normalize Data
    X_train = NormalizeData(X_train, normRange=(0.0, 1.0))
    X_test = NormalizeData(X_test, normRange=(0.0, 1.0))
    # Split Data
    X_train_classwise = SplitDataset_ClassWise(X_train, Y_train)
    X_test_classwise = SplitDataset_ClassWise(X_test, Y_test)
    # Flatten Input and One Hot Output
    X_train_flat = FlattenData(X_train)
    X_test_flat = FlattenData(X_test)
    Y_train_oh = OneHotEncoder(Y_train, DATASET_N_CLASSES)
    Y_test_oh = OneHotEncoder(Y_test, DATASET_N_CLASSES)
    # Print
    if VERBOSE:
        print("Train:")
        print("X_train_classwise:", [len(X_train_classwise[i]) for i in range(DATASET_N_CLASSES)])
        print("X_train:", X_train.shape, X_train.min(), X_train.max())
        print("Y_train:", Y_train.shape)
        print("X_train_flat:", X_train_flat.shape, X_train_flat.min(), X_train_flat.max())
        print("Y_train_oh:", Y_train_oh.shape)
        print()
        print("Test:")
        print("X_test_classwise:", [len(X_test_classwise[i]) for i in range(DATASET_N_CLASSES)])
        print("X_test:", X_test.shape, X_test.min(), X_test.max())
        print("Y_test:", Y_test.shape)
        print("X_test_flat:", X_test_flat.shape, X_test_flat.min(), X_test_flat.max())
        print("Y_test_oh:", Y_test_oh.shape)
        print()

    FUNCS = {
        # Evaluation Function
        "eval_fn": {
            "func": EvalFunc_Accuracy,
            "params": {},
        }
    }

    # Test
    X_test = X_test_flat
    Y_test = Y_test_oh
    # Test Loss
    W_flat = np.concatenate([w.flatten() for w in TRAINED_PARAMETERS["Ws"]])
    b_flat = np.concatenate([b.flatten() for b in TRAINED_PARAMETERS["bs"]])
    params_flat = np.concatenate([W_flat, b_flat])
    loss_reg = TRAINED_PARAMETERS["loss_fn"]["params"]["l2_lambda"] * np.sqrt(np.sum(params_flat**2)) + \
                TRAINED_PARAMETERS["loss_fn"]["params"]["l1_lambda"] * np.sum(np.abs(params_flat))
    Y_test_out, As, Os = ForwardPropogation(X_test, TRAINED_PARAMETERS)
    loss_test = loss_reg + TRAINED_PARAMETERS["loss_fn"]["func"](Y_test_out.T, Y_test, **TRAINED_PARAMETERS["loss_fn"]["params"])
    print("Test Loss:", loss_test)
    # Test Eval
    Y_test_pred = np.argmax(Y_test_out.T, axis=-1)
    Y_test = np.argmax(Y_test, axis=-1)
    eval_test = FUNCS["eval_fn"]["func"](Y_test_pred, Y_test, **FUNCS["eval_fn"]["params"])
    print("Test Eval:", eval_test, "/", Y_test.shape[0])
    
# Run
if __name__ == "__main__":
    # Parse Arguments
    args = Runner_ParseArgs()
    # Run
    if args.mode == "train":
        Runner_Train(args)
    elif args.mode == "test":
        Runner_Test(args)
    else:
        print("Invalid Mode.")