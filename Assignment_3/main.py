'''
Main
'''

# Imports
import json
import argparse
import functools
from pprint import pprint

from Model import *
from Visualise_AttnHeatMap import *
from Visualise_Connectivity import *

# Main Classes
class CONFIG:
    '''
    Config Class
    '''
    def __init__(self, **params):
        self.__dict__.update(params)

# Main Functions
# Run Function
def Model_Sweep_Run(wandb_data, config=None):
    '''
    Model Sweep Runner
    '''
    # Init
    if wandb_data["enable"]: wandb.init()

    # Get Run Config
    config = wandb.config if wandb_data["enable"] else config
    N_EPOCHS = config.n_epochs
    BATCH_SIZE = config.batch_size

    ENCODER = config.encoder
    DECODER = ENCODER
    ENCODER_EMBEDDING_SIZE = config.encoder_embedding_size
    DECODER_EMBEDDING_SIZE = ENCODER_EMBEDDING_SIZE
    ENCODER_N_UNITS = config.encoder_n_units
    DECODER_N_UNITS = ENCODER_N_UNITS
    ACT_FUNC = config.act_func
    DROPOUT = config.dropout
    LOSS_FN = wandb_data["loss_fn"]
    USE_ATTENTION = wandb_data["attention"]
    ATTENTION_N_UNITS = ENCODER_N_UNITS[-1]

    print("RUN CONFIG:")
    pprint(config)
    print("OTHER CONFIG:")
    pprint(wandb_data)

    # Get Inputs
    inputs = {
        "model": {
            "blocks": {
                "encoder": [
                    functools.partial(BLOCKS_ENCODER[ENCODER], 
                        n_units=ENCODER_N_UNITS[i], activation=ACT_FUNC, 
                        dropout=DROPOUT, recurrent_dropout=DROPOUT, 
                        return_state=True, return_sequences=(i < (len(ENCODER_N_UNITS)-1)) or USE_ATTENTION, 
                    ) for i in range(len(ENCODER_N_UNITS))
                ],
                "decoder": [
                    functools.partial(BLOCKS_DECODER[DECODER], 
                        n_units=DECODER_N_UNITS[i], activation=ACT_FUNC, 
                        dropout=DROPOUT, recurrent_dropout=DROPOUT, 
                        return_state=True, return_sequences=True, 
                    ) for i in range(len(DECODER_N_UNITS))
                ],
            }, 
            "compile_params": {
                "loss_fn": LOSS_FUNCTIONS[LOSS_FN](),
                "optimizer": Adam(),
                "metrics": ["accuracy"]
            }
        }
    }

    # Get Train Val Dataset
    DATASET, DATASET_ENCODED = LoadTrainDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    inputs["dataset_encoded"] = DATASET_ENCODED
    inputs["dataset_encoded"]["train"]["batch_size"] = BATCH_SIZE
    inputs["dataset_encoded"]["val"]["batch_size"] = BATCH_SIZE

    # Build Model
    X_shape = DATASET_ENCODED["train"]["encoder_input"].shape
    Y_shape = DATASET_ENCODED["train"]["decoder_output"].shape
    MODEL = Model_EncoderDecoderBlocks(
        X_shape=X_shape, Y_shape=Y_shape, 
        Blocks=inputs["model"]["blocks"],
        encoder={
            "embedding_size": ENCODER_EMBEDDING_SIZE
        }, 
        decoder={
            "embedding_size": DECODER_EMBEDDING_SIZE
        },
        use_attention=USE_ATTENTION,
        attn_n_units=ATTENTION_N_UNITS
    )
    MODEL = Model_Compile(MODEL, **inputs["model"]["compile_params"])

    # Train Model
    TRAINED_MODEL, TRAIN_HISTORY = Model_Train(
        MODEL, inputs, N_EPOCHS, wandb_data, 
        best_model_path=PATH_BESTMODEL
    )

    # Load Best Model
    TRAINED_MODEL = Model_LoadModel(PATH_BESTMODEL)
    # Get Test Dataset
    DATASET_TEST, DATASET_ENCODED_TEST = LoadTestDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    # Test Best Model
    loss_test, eval_test, eval_test_inference = Model_Test(
        TRAINED_MODEL, DATASET_ENCODED_TEST,
        target_chars=DATASET_ENCODED_TEST["chars"]["target_chars"],
        target_char_map=DATASET_ENCODED_TEST["chars"]["target_char_map"],
        dataset_words=DATASET_TEST
    )

    print("MODEL TEST:")
    print("Loss:", loss_test)
    print("Eval:", eval_test)
    print("Eval Inference:", eval_test_inference)

    # Wandb log test data
    if wandb_data["enable"]:
        wandb.log({
            "loss_test": loss_test,
            "eval_test": eval_test,
            "eval_test_inference": eval_test_inference
        })

        # Close Wandb Run
        wandb.finish()

# Runner Functions
def Runner_ParseArgs():
    '''
    Parse Args
    '''
    global DATASET_PATH_DAKSHINA_TAMIL
    
    parser = argparse.ArgumentParser(description="Training and Testing for DL Assignment 3")

    parser.add_argument("--mode", "-m", type=str, default="train", help="train | test | vh | vc")
    parser.add_argument("--model", "-ml", type=str, default="Models/Model.h5", help="Model path to use or save to")
    parser.add_argument("--dataset", "-dt", type=str, default=DATASET_PATH_DAKSHINA_TAMIL, help="Dataset path to use")

    # Train Args
    parser.add_argument("--epochs", "-e", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")

    parser.add_argument("--encoder", "-en", type=str, default="LSTM", help="Encoder type")
    # parser.add_argument("--decoder", "-de", type=str, default="LSTM", help="Decoder type")
    parser.add_argument("--encoder_embedding_size", "-es", type=int, default=2, help="Encoder embedding size")
    # parser.add_argument("--decoder_embedding_size", "-des", type=int, default=2, help="Decoder embedding size")
    parser.add_argument("--encoder_n_units", "-eu", type=str, default="2", help="Encoder Num units")
    # parser.add_argument("--decoder_n_units", "-du", type=str, default="2", help="Decoder Num units")
    parser.add_argument("--act_func", "-af", type=str, default="tanh", help="Activation function")
    parser.add_argument("--dropout", "-d", type=float, default=0.2, help="Dropout")
    parser.add_argument("--attention", "-a", action="store_true", help="Use Attention Layer")

    args = parser.parse_args()
    DATASET_PATH_DAKSHINA_TAMIL = str(args.dataset).rstrip("/") + "/ta/lexicons/ta.translit.sampled.{}.tsv"
    return args

def Runner_Train(args):
    '''
    Train Model
    '''
    # Load Wandb Data
    WANDB_DATA = json.load(open("config.json", "r"))
    WANDB_DATA.update({
        "attention": args.attention,
        "loss_fn": "categorical_crossentropy" # "categorical_crossentropy", sparse_categorical_crossentropy"
    })
    SWEEP_CONFIG = {}
    if WANDB_DATA["enable"]:
        # Sweep Setup - WANDB
        SWEEP_CONFIG = {
            "name": "lstm-run-1",
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

                "encoder": {
                    "values": [args.encoder]
                },
                "encoder_embedding_size": {
                    "values": [args.encoder_embedding_size]
                },
                "encoder_n_units": {
                    "values": [
                        [int(x) for x in args.encoder_n_units.split(",")]
                    ]
                },
                "act_func": {
                    "values": [args.act_func]
                },
                "dropout": {
                    "values": [args.dropout]
                }
            }
        }
        # Run Sweep
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"])
        # sweep_id = ""
        TRAINER_FUNC = functools.partial(Model_Sweep_Run, wandb_data=WANDB_DATA)
        wandb.agent(sweep_id, TRAINER_FUNC, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"], count=1)

    else:
        # Sweep Setup - Non-WANDB
        SWEEP_CONFIG = {
                "n_epochs": args.epochs,
                "batch_size": args.batch_size,

                "encoder": args.encoder,
                "encoder_embedding_size": args.encoder_embedding_size,
                "encoder_n_units": [int(x) for x in args.encoder_n_units.split(",")],

                "act_func": args.act_func,
                "dropout": args.dropout,
        }
        config = CONFIG(**SWEEP_CONFIG)
        # Run
        Model_Sweep_Run(WANDB_DATA, config=config)

    # Save Model
    Model_SaveModel(Model_LoadModel(PATH_BESTMODEL), args.model)

def Runner_Test(args):
    '''
    Test Model
    '''
    # Load Model
    TRAINED_MODEL = Model_LoadModel(args.model)
    # Get Test Dataset
    DATASET_TEST, DATASET_ENCODED_TEST = LoadTestDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    # Test Best Model
    loss_test, eval_test, eval_test_inference = Model_Test(
        TRAINED_MODEL, DATASET_ENCODED_TEST,
        target_chars=DATASET_ENCODED_TEST["chars"]["target_chars"],
        target_char_map=DATASET_ENCODED_TEST["chars"]["target_char_map"],
        dataset_words=DATASET_TEST
    )
    # Display
    print("MODEL TEST:")
    print("Loss:", loss_test)
    print("Accuracy:", eval_test)
    print("Accuracy Inference:", eval_test_inference)

def Runner_VisAttnHeatMap(args):
    '''
    Visualize Attention HeatMap
    '''
    # Load Model
    TRAINED_MODEL = Model_LoadModel(args.model)
    # Get Test Dataset
    DATASET_TEST, DATASET_ENCODED_TEST = LoadTestDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    # Plot HeatMap
    Vis_AttnHeatMap(TRAINED_MODEL, DATASET_TEST, DATASET_ENCODED_TEST, 9)

def Runner_VisConnectivity(args):
    '''
    Visualize Connectivity
    '''
    # Load Model
    TRAINED_MODEL = Model_LoadModel(args.model)
    # Get Test Dataset
    DATASET_TEST, DATASET_ENCODED_TEST = LoadTestDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    # Plot Connectivity
    Vis_Connectivity(TRAINED_MODEL, DATASET_TEST, DATASET_ENCODED_TEST, 1)

# Run
if __name__ == "__main__":
    # Parse Args
    ARGS = Runner_ParseArgs()
    # Run
    if ARGS.mode == "train":
        Runner_Train(ARGS)
    elif ARGS.mode == "test":
        Runner_Test(ARGS)
    elif ARGS.mode == "vh":
        Runner_VisAttnHeatMap(ARGS)
    elif ARGS.mode == "vc":
        Runner_VisConnectivity(ARGS)
    else:
        print("Invalid Mode!")