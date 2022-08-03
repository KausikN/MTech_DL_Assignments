'''
Model
'''

# Imports
import wandb
from wandb.keras import WandbCallback
import math
from keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras as TFKeras
from tqdm import tqdm
from tensorflow.keras.utils import plot_model

from Library.BahdanauAttention import *
from Library.ModelBlocks import *
from Dataset import *

# Main Vars
PATH_BESTMODEL = "Models/best_model.h5"
LOSS_FUNCTIONS = {
    "categorical_crossentropy": CategoricalCrossentropy,
    "sparse_categorical_crossentropy": SparseCategoricalCrossentropy
}

# Main Functions
# Model Functions
# Build Encoder-Decoder Model Function
def Model_EncoderDecoderBlocks(X_shape, Y_shape, Blocks, **params):
    '''
    Encoder Decoder Model
    '''
    # Encoder
    print("Encoder")
    # Input Layer
    print("X_shape:", X_shape)
    print("Y_shape:", Y_shape)
    encoder_input = Input(shape=(None,), name="encoder_input")
    print("Encoder Input:", encoder_input.shape)
    encoder_embedding = Embedding(
        DATASET_DAKSHINA_TAMIL_UNIQUE_CHARS["input"]+1, params["encoder"]["embedding_size"], 
        # input_length=X_shape[1], 
        mask_zero=True, 
        name="encoder_embedding"
    )(encoder_input)
    print("Encoder Embedding:", encoder_embedding.shape)
    encoder_outputs = encoder_embedding
    # Add Blocks
    encoderData = []
    for i in range(len(Blocks["encoder"])):
        encoderData.append(Blocks["encoder"][i](encoder_outputs, block_name="encoder_block_" + str(i)))
        encoder_outputs = encoderData[-1]["output"]
        print("Encoder Block:", i, encoder_outputs.shape)

    print()
    print("Decoder")
    
    # Decoder
    # Input Layer
    decoder_input = Input(shape=(None,), name="decoder_input")
    print("Decoder Input:", decoder_input.shape)
    decoder_embedding = Embedding(
        DATASET_DAKSHINA_TAMIL_UNIQUE_CHARS["target"]+1, params["decoder"]["embedding_size"],
        # input_length=Y_shape[1], 
        mask_zero=True,
        name="decoder_embedding"
    )(decoder_input)
    print("Decoder Embedding:", decoder_embedding.shape)
    decoder_outputs = decoder_embedding
    # Add Blocks
    decoderData = []
    for i in range(len(Blocks["decoder"])):
        initial_state = encoderData[i]["state"]
        decoderData.append(Blocks["decoder"][i](decoder_outputs, initial_state, block_name="decoder_block_" + str(i)))
        decoder_outputs = decoderData[-1]["output"]
        print("Decoder Block:", i, decoder_outputs.shape)
    
    if params["use_attention"]:
        # Attention Layer
        print("Encoder Output:", encoderData[-1]["output"])
        print("Decoder Output:", decoderData[-1]["output"])

        zero = tf.constant(0, dtype=tf.float32)
        mask = tf.not_equal(encoder_input, zero)
        print("Mask:", mask.shape)

        att_output, att_states = BahdanauAttention(params["attn_n_units"], name="attention")(
            query=decoderData[-1]["output"],
            value=encoderData[-1]["output"],
            mask=mask
        )
        print("Attention Output:", att_output.shape)
        # Concat Layer
        decoder_concat_input = tf.concat([att_output, decoderData[-1]["output"]], axis=-1)
        print("Decoder Concat Input:", decoder_concat_input.shape)
        att_vector = decoder_concat_input
        print("Attention Vector:", att_vector.shape)
        # Output Layer
        decoder_outputs = TimeDistributed(Dense(
            DATASET_DAKSHINA_TAMIL_UNIQUE_CHARS["target"]+1, activation="softmax"
        ), name="decoder_dense")(att_vector)
    else:
        # Output Layer
        decoder_outputs = Dense(
            DATASET_DAKSHINA_TAMIL_UNIQUE_CHARS["target"]+1, activation="softmax", name="decoder_dense"
        )(decoderData[-1]["output"])

    print("Decoder Output:", decoder_outputs.shape)

    # Construct Model
    model = Model([encoder_input, decoder_input], decoder_outputs)

    return model

# Common Functions
# Compile Model Function
def Model_Compile(model, loss_fn="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], **params):
    '''
    Compile Model
    '''
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    return model

# Train Model Function
def Model_Train(model, inputs, n_epochs, wandb_data, **params):
    '''
    Train Model
    '''
    OUTPUT_LABEL_ENCODING = (model.loss.name == "sparse_categorical_crossentropy")
    # Get Data
    DATASET_ENCODED = inputs["dataset_encoded"]
    dataset_train_encoder_input = np.argmax(DATASET_ENCODED["train"]["encoder_input"], axis=-1)
    dataset_val_encoder_input = np.argmax(DATASET_ENCODED["val"]["encoder_input"], axis=-1)
    dataset_train_decoder_input = np.argmax(DATASET_ENCODED["train"]["decoder_input"], axis=-1)
    dataset_val_decoder_input = np.argmax(DATASET_ENCODED["val"]["decoder_input"], axis=-1)
    
    if OUTPUT_LABEL_ENCODING:
        dataset_train_decoder_output = np.argmax(DATASET_ENCODED["train"]["decoder_output"], axis=-1)
        dataset_val_decoder_output = np.argmax(DATASET_ENCODED["val"]["decoder_output"], axis=-1)
    else:
        dataset_train_decoder_output = DATASET_ENCODED["train"]["decoder_output"]
        dataset_val_decoder_output = DATASET_ENCODED["val"]["decoder_output"]
    
    TRAIN_STEP_SIZE = math.ceil(dataset_train_encoder_input.shape[0] / DATASET_ENCODED["train"]["batch_size"])
    VALIDATION_STEP_SIZE = math.ceil(dataset_val_encoder_input.shape[0] / DATASET_ENCODED["val"]["batch_size"])

    callbacks = []
    # Enable Wandb Callback
    if wandb_data["enable"]:
        WandbCallbackFunc = WandbCallback(
            monitor="val_accuracy", save_model=True, log_evaluation=True, log_weights=True,
            log_best_prefix="best_",
            validation_steps=VALIDATION_STEP_SIZE
        )
        callbacks.append(WandbCallbackFunc)
    # Enable Model Checkpointing
    ModelCheckpointFunc = ModelCheckpoint(
        params["best_model_path"],
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
        save_freq="epoch"
    )
    callbacks.append(ModelCheckpointFunc)

    # Train Model
    TRAIN_HISTORY = model.fit(
        [dataset_train_encoder_input, dataset_train_decoder_input], dataset_train_decoder_output, 
        steps_per_epoch=TRAIN_STEP_SIZE,
        validation_data=([dataset_val_encoder_input, dataset_val_decoder_input], dataset_val_decoder_output), 
        validation_steps=VALIDATION_STEP_SIZE,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks
    )

    return model, TRAIN_HISTORY

# Test Model Function
def Model_Test(model, dataset, **params):
    '''
    Test Model
    '''
    # Plot Model Data
    print(model.summary())
    # plot_model(model, to_file="Outputs/model_plot.png", show_shapes=True, show_layer_names=True)
    # Get Data
    OUTPUT_LABEL_ENCODING = (model.loss.name == "sparse_categorical_crossentropy")
    dataset_test_encoder_input = np.argmax(dataset["encoder_input"], axis=-1)
    dataset_test_decoder_input = np.argmax(dataset["decoder_input"], axis=-1)
    if OUTPUT_LABEL_ENCODING:
        dataset_test_decoder_output = np.argmax(dataset["decoder_output"], axis=-1)
    else:
        dataset_test_decoder_output = dataset["decoder_output"]
    print(OUTPUT_LABEL_ENCODING)
    print(dataset_test_encoder_input.shape)
    print(dataset_test_decoder_input.shape)
    print(dataset_test_decoder_output.shape)
    
    # Test Model - Charecter Level
    loss_test, eval_test = model.evaluate(
        [dataset_test_encoder_input, dataset_test_decoder_input], 
        dataset_test_decoder_output, 
        verbose=1
    )
    # Test Model - Word Level
    # Predict Output for each word
    params["use_attention"] = "attention" in [layer.name for layer in model.layers]
    encoder_model, decoder_model = Model_Inference_GetEncoderDecoder(model, **params)

    outputs = []
    n = dataset_test_encoder_input.shape[0]
    batch_size = 128
    for i in tqdm(range(0, n, batch_size)):
        # Inputs
        end_i = min(i + batch_size, n)
        words = dataset_test_encoder_input[i:end_i]
        # Results
        decoded_words, _ = Model_Inference_Transliterate(words, encoder_model, decoder_model, **params)
        outputs = outputs + decoded_words

    # Remove SOS and EOS
    target_words = dataset_test_decoder_output
    if not OUTPUT_LABEL_ENCODING: target_words = np.argmax(dataset_test_decoder_output, axis=-1)
    # Get Charecter String
    target_words = ["".join([params["target_chars"][ci] for ci in word]) for word in target_words]
    # Remove SOS and EOS
    target_words = [word[:word.find(SYMBOLS["end"])] for word in target_words]

    # Evaluations
    # Size Matches
    sizeMatches = np.count_nonzero([len(target_words[i]) == len(outputs[i]) for i in range(len(target_words))])
    print("Size Matches:", sizeMatches, "/", len(target_words), "=", sizeMatches / len(target_words))
    # Calculate Word Level Accuracy
    outputs = np.array(outputs)
    target_words = np.array(target_words)
    print("True:", target_words)
    print("Predicted:", outputs)
    output_shapes = [len(w) for w in outputs]
    print("Predicted Sizes (Unique):", np.unique(output_shapes))
    print("Predicted Sizes:", np.array(output_shapes))
    eval_test_inference = np.mean(outputs == target_words)

    # Form Predictions CSV
    if "dataset_words" in params.keys():
        predictions = {
            "english_word": params["dataset_words"]["input"],
            "true_tamil_word": params["dataset_words"]["target"],
            "predicted_tamil_word": outputs
        }
        predictions_df = pd.DataFrame(predictions)
        fname = "vanilla" if not params["use_attention"] else "attention"
        predictions_df.to_csv(f"predictions_{fname}/predictions_{fname}.csv", index=False)

    return loss_test, eval_test, eval_test_inference

# Load and Save Model Functions
def Model_LoadModel(path):
    '''
    Load Model
    '''
    return load_model(path, custom_objects={"BahdanauAttention": BahdanauAttention})

def Model_SaveModel(model, path):
    '''
    Save Model
    '''
    return model.save(path)

# Inference Functions
def Model_Inference_GetEncoderDecoder(model, **params):
    '''
    Get Encoder and Decoder for Inferencing Model
    '''
    # Encoder Input
    encoder_inputs = model.input[0]

    # Encoder Outputs
    encoder_data = {}
    encoder_states = []
    for layer in model.layers:
        name = str(layer.name)
        if name.startswith("encoder_block_"):
            li = int(name.lstrip("encoder_block_").split("_")[0])
            out_data = layer.output
            encoder_data[li] = out_data

    if params["use_attention"]: encoder_states.append(encoder_data[list(sorted(encoder_data.keys()))[-1]][0])
    for i in list(sorted(encoder_data.keys())):
        encoder_states.extend(encoder_data[i][1:])

    # Get Encoder Model which gives cell states as output
    model_encoder = Model(encoder_inputs, encoder_states)

    print("Encoder Outputs:", len(encoder_states))

    # Decoder Input
    decoder_inputs = model.input[1]

    # Decoder Outputs
    decoder_data = {}
    decoders = []
    decoder_states = []
    mask_layer = None
    attention_layer, concat_layer, attention_dense_layer, decoder_hidden_state_inputs = None, None, None, None
    decoder_dense, decoder_embedding_layer = None, None
    for layer in model.layers:
        if layer.name =="decoder_dense":
            decoder_dense = layer
        elif layer.name == "decoder_embedding":
            decoder_embedding_layer = layer
        elif layer.name == "tf.math.not_equal":
            mask_layer = layer
        elif layer.name == "attention":
            attention_layer = layer
            decoder_hidden_state_inputs = Input(shape=(None, n_cells))
        elif layer.name in ["concat", "tf.concat"]:
            concat_layer = layer
        elif layer.name == "attention_vector":
            attention_dense_layer = layer
        else:
            name = str(layer.name)
            if name.startswith("decoder_block_"):
                li = int(name.lstrip("decoder_block_").split("_")[0])
                n_cells = layer.output_shape[0][-1]
                decoder_data[li] = [layer]
                state_count = len(encoder_data[li])-1
                for i in range(state_count):
                    decoder_data[li].append(Input(shape=(n_cells,)))
    # Decoder Layers
    for i in list(sorted(decoder_data.keys())):
        decoders.append(decoder_data[i][0])
        decoder_states.append(decoder_data[i][1:])

    decoder_outputs = decoder_embedding_layer(decoder_inputs)
    decoder_states_outputs = []

    for i in range(len(decoders)):
        out_data = decoders[i](decoder_outputs, initial_state=decoder_states[i])
        decoder_outputs = out_data[0]
        decoder_states_outputs.extend(out_data[1:])

    if params["use_attention"]:
        # Attention Layer
        att_output_inf, att_states_inf = attention_layer(
            query=decoder_outputs, 
            value=decoder_hidden_state_inputs
        )
        decoder_states_outputs.append(att_states_inf)
        # Concat Layer
        decoder_concat_output = concat_layer([att_output_inf, decoder_outputs], axis=-1)
        # Dense Layer
        decoder_outputs = decoder_concat_output

    # Softmax layer
    decoder_outputs = decoder_dense(decoder_outputs)
    # Create the decoder model
    decoder_states_inputs = []
    for i in range(len(decoder_states)): decoder_states_inputs.extend(decoder_states[i])
    di = [decoder_inputs]
    if params["use_attention"]: di.append(decoder_hidden_state_inputs)
    for s in decoder_states_inputs: di.append(s)
    do = [decoder_outputs]
    for s in decoder_states_outputs: do.append(s)
    model_decoder = Model(di, do)

    print("Decoder Inputs:", len(di))
    print("Decoder Outputs:", len(do))
    
    return model_encoder, model_decoder

def Model_Inference_Transliterate(words, model_encoder, model_decoder, **params):
    '''
    Transliterate Input Words
    '''
    batch_size = words.shape[0]
    # Encode the input string
    encoded_states = model_encoder.predict(words)

    target_sequence = np.zeros((batch_size, 1, DATASET_DAKSHINA_TAMIL_UNIQUE_CHARS["target"]+1))
    # Set SOS
    target_sequence[:, 0, params["target_char_map"][SYMBOLS["start"]]] = 1.0
    target_sequence = np.argmax(target_sequence, axis=-1)

    decoded_words = [""]*batch_size
    attentions = []
    for i in range(DATASET_DAKSHINA_TAMIL_MAX_CHARS["target"]):
        decoder_inputs = [target_sequence]
        for s in encoded_states: decoder_inputs.append(s)

        decoded_data = model_decoder.predict(decoder_inputs)
        output_tokens = decoded_data[0]
        decoded_states = decoded_data[1:]
        if params["use_attention"]:
            attentions.append(decoded_data[-1])
            decoded_states = decoded_states[:-1] # Ignore the Attn States

        # Get predicted character and update target sequence
        char_pred = np.argmax(output_tokens[:, -1, :], axis=-1)
        target_sequence = np.zeros((batch_size, 1, DATASET_DAKSHINA_TAMIL_UNIQUE_CHARS["target"]+1))
        for j in range(char_pred.shape[0]):
            ci = char_pred[j]
            decoded_words[j] += params["target_chars"][ci]
            target_sequence[j, 0, ci] = 1.0
        target_sequence = np.argmax(target_sequence, axis=-1)
        # Update states
        encoded_states_new = []
        if params["use_attention"]: encoded_states_new = [encoded_states[0]]
        encoded_states = encoded_states_new + decoded_states

    # Remove EOS
    decoded_words = [word[:word.find(SYMBOLS["end"])] for word in decoded_words]
    
    return decoded_words, attentions