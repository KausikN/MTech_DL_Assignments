'''
Question 2 to 10
'''

# Imports
from pprint import pprint
import json
import functools

from Dataset import *
from FNN import *

# Main Functions
def Model_Sweep_Run(inputs={}, funcs={}, wandb_data={"use_wandb": False}):
    wandb.init(project=wandb_data["project_name"], entity=wandb_data["user_name"])
    config = wandb.config
    X, Y, X_val, Y_val, X_test, Y_test = inputs["X"], inputs["Y"], inputs["X_val"], inputs["Y_val"], inputs["X_test"], inputs["Y_test"]

    # Get Run Config
    N_EPOCHS = config.N_EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    N_HIDDEN_LAYERS = config.N_HIDDEN_LAYERS
    N_HIDDEN_NEURONS = config.N_HIDDEN_NEURONS
    WEIGHT_DECAY = config.WEIGHT_DECAY
    LEARNING_RATE = config.LEARNING_RATE
    OPTIMISER_NAME = config.OPTIMISER
    INIT_FUNC_NAME = config.INIT_FUNC
    ACT_FUNC_NAME = config.ACT_FUNC
    OUT_ACT_FUNC_NAME = config.OUT_ACT_FUNC

    HIDDEN_NETWORK_LAYERS = [N_HIDDEN_NEURONS] * N_HIDDEN_LAYERS
    NETWORK_LAYERS = [X.shape[1]] + HIDDEN_NETWORK_LAYERS + [Y.shape[1]]

    ACT_FUNC = ACTIVATION_FUNCTIONS[ACT_FUNC_NAME]
    OUT_ACT_FUNC = ACTIVATION_FUNCTIONS[OUT_ACT_FUNC_NAME]
    ACT_FUNC["params"] = {}
    OUT_ACT_FUNC["params"] = {}

    LOSS_FUNC = funcs["loss_fn"]
    LOSS_FUNC["params"].update({
        "l1_lambda": 0.0,
        "l2_lambda": WEIGHT_DECAY
    })

    OPTIMISER_FUNC = OPTIMISERS[OPTIMISER_NAME]
    OPTIMISER = {
        "func": OPTIMISER_FUNC,
        "params": {
            "lr": LEARNING_RATE,
            "gamma": 0.5,
            "eps": 1e-8,
            "beta1": 0.2,
            "beta2": 0.2
        },
        "data": {}
    }

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
        "loss_fn": LOSS_FUNC,

        # Update Function
        "update_fn": OPTIMISER,

        # Evaluation Function
        "eval_fn": funcs["eval_fn"]
    }

    print("RUN CONFIG:")
    pprint(config)
    print("FUNCS:")
    pprint(FUNCS)

    # Train Model
    TRAINED_PARAMETERS, TRAIN_HISTORY = Model_Train(inputs, NETWORK_LAYERS, FUNCS, N_EPOCHS, BATCH_SIZE, wandb_data)

    # Test
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
    print("Test Eval:", eval_test)

    # Wandb log test data
    wandb.log({
        "loss_test": loss_test,
        "eval_test": eval_test
    })

    # Close Wandb Run
    wandb.finish()

# Run
# Params
WANDB_DATA = json.load(open("config.json", "r"))
WANDB_DATA.update({
    "use_wandb": True
})

# HIDDEN_NETWORK_LAYERS = [128, 16] # Input and Output layer sizes are added at beggining and end automatically
FUNCS = {
    # Loss Function
    "loss_fn": {
        "func": LossFunc_CrossEntropy,
        "deriv": LossFuncDeriv_CrossEntropy_WithSoftmax,
        "params": {}
    },

    # Evaluation Function
    "eval_fn": {
        "func": EvalFunc_Accuracy,
        "params": {},
    }
}
# Params

# Sweep Setup
SWEEP_CONFIG = {
    "name": "momentum-sweep-2",
    "method": "random",
    "metric": {
        "name": "loss_val",
        "goal": "minimize"
    },
    "parameters": {
        "N_EPOCHS": {
            "values": [1]
        },
        "BATCH_SIZE": {
            "values": [32, 64, 128]
        },
        "N_HIDDEN_LAYERS": {
            "values": [5]
        },
        "N_HIDDEN_NEURONS": {
            "values": [64, 128]
        },
        "WEIGHT_DECAY": {
            "values": [0.0005, 0.005]
        },
        "LEARNING_RATE": {
            "values": [5e-3, 1e-3, 5e-4]
        },
        "OPTIMISER": {
            "values": ["momentum"] # ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]
        },
        "INIT_FUNC": {
            "values": ["xavier"] # ["random", "xavier"]
        },
        "ACT_FUNC": {
            "values": ["tanh", "relu"] # ["sigmoid", "tanh", "relu"]
        },
        "OUT_ACT_FUNC": {
            "value": "softmax"
        }
    }
}

# Run
# Load Data
X_train_full, X_test, Y_train, Y_test = LoadFashionDataset()
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

# Run
INPUTS = {
    "X": X_train_flat,
    "Y": Y_train_oh,
    "X_val": X_val_flat,
    "Y_val": Y_val_oh,
    "X_test": X_test_flat,
    "Y_test": Y_test_oh
}
# Init Wandb Sweep
# sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"])
sweep_id = "voiqaeon"
TRAINER_FUNC = functools.partial(Model_Sweep_Run, inputs=INPUTS, funcs=FUNCS, wandb_data=WANDB_DATA)
wandb.agent(sweep_id, TRAINER_FUNC, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"], count=1)