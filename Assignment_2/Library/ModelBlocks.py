'''
Model Block Functions
'''

# Imports
from keras.layers import *

# Main Functions
# Block Functions
# Convolution-ReLu-MaxPool Block @ Karthikeyan S CS21M028
def Block_CRM(
    model, inp_shape, 
    conv_filters=32, conv_kernel_size=(3, 3), 
    batch_norm=True, 
    act_fn="relu", 
    maxpool_pool_size=(2, 2), maxpool_strides=(2, 2), 
    dropout_rate=0.1, 
    block_name="0", **params
    ):
    '''
    Conv -> Relu -> MaxPool Block
    '''
    # Conv Layer
    model.add(Conv2D(conv_filters, conv_kernel_size, input_shape=inp_shape, name=block_name + "_conv"))
    # BatchNorm Layer
    if batch_norm: model.add(BatchNormalization(name=block_name + "_batchnorm"))
    # Act Layer
    model.add(Activation(act_fn, name=block_name + "_act"))
    # MaxPool Layer
    model.add(MaxPooling2D(pool_size=maxpool_pool_size, strides=maxpool_strides, name=block_name + "_maxpool"))
    # Dropout Layer
    if dropout_rate > 0.0: model.add(Dropout(dropout_rate, name=block_name + "_dropout"))

    output_shape = model.get_layer(block_name + "_maxpool").output_shape
    return model, output_shape
    
# Main Vars
BLOCKS = {
    "CRM": Block_CRM
}