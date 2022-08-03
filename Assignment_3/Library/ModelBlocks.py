'''
Model Block Functions
'''

# Imports
from tensorflow.keras.layers import *

# Main Functions
# Encoder Block Functions
# RNN Block
def EncoderBlock_RNN(
    model, 
    n_units=256, 
    activation="relu", 
    dropout=0.1, recurrent_dropout=0.1, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    RNN Block
    '''
    encoder = SimpleRNN(
        n_units, 
        activation=activation, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_RNN"
    )
    data = encoder(model)
    encoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return encoder_data

# LSTM Block
def EncoderBlock_LSTM(
    model, 
    n_units=256, 
    activation="relu", 
    dropout=0.1, recurrent_dropout=0.1, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    LSTM Block
    '''
    encoder = LSTM(
        n_units, 
        activation=activation, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_LSTM"
    )
    data = encoder(model)
    encoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return encoder_data

# GRU Block
def EncoderBlock_GRU(
    model, 
    n_units=256, 
    activation="relu", 
    dropout=0.1, recurrent_dropout=0.1, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    GRU Block
    '''
    encoder = GRU(
        n_units, 
        activation=activation, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_GRU"
    )
    data = encoder(model)
    encoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return encoder_data

# Decoder Block Functions
# RNN Block
def DecoderBlock_RNN(
    model, initial_state, 
    n_units=256, 
    activation="relu", 
    dropout=0.1, recurrent_dropout=0.1, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    RNN Block
    '''
    decoder = SimpleRNN(
        n_units, 
        activation=activation, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_RNN"
    )
    data = decoder(model, initial_state=initial_state)
    decoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return decoder_data

# LSTM Block
def DecoderBlock_LSTM(
    model, initial_state, 
    n_units=256, 
    activation="relu", 
    dropout=0.1, recurrent_dropout=0.1, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    LSTM Block
    '''
    decoder = LSTM(
        n_units, 
        activation=activation, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_LSTM"
    )
    data = decoder(model, initial_state=initial_state)
    decoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return decoder_data

# GRU Block
def DecoderBlock_GRU(
    model, initial_state, 
    n_units=256, 
    activation="relu", 
    dropout=0.1, recurrent_dropout=0.1, 
    return_state=False, return_sequences=False, 
    block_name="0", **params
    ):
    '''
    GRU Block
    '''
    decoder = GRU(
        n_units, 
        activation=activation, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, 
        return_state=return_state, return_sequences=return_sequences, 
        name=block_name + "_GRU"
    )
    data = decoder(model, initial_state=initial_state)
    decoder_data = {
        "output": data[0],
        "state": data[1:]
    }

    return decoder_data
    
# Main Vars
BLOCKS_ENCODER = {
    "RNN": EncoderBlock_RNN,
    "LSTM": EncoderBlock_LSTM,
    "GRU": EncoderBlock_GRU
}
BLOCKS_DECODER = {
    "RNN": DecoderBlock_RNN,
    "LSTM": DecoderBlock_LSTM,
    "GRU": DecoderBlock_GRU
}