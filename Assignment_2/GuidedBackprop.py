'''
Guided Backpropogation
'''

# Imports
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

import matplotlib.pyplot as plt
from Model import *

# Main Functions
# Replace activations by guidedbackprop activations
@tf.custom_gradient
def ReLu_GuidedBackProp(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad

# Guided Backprop on a input model @ N Kausik CS21M037
def GuidedBackprop_Display(MODEL, X_shape=(227, 227, 3), Y_shape=10, nCols=2, dataset_path=DATASET_PATH_INATURALIST):
    '''
    Display Guided Backpropogation on model
    '''
    # Load Random Dataset Image
    I_path, I_class = GetImagePath_Random(dataset_path)
    I = tf.keras.preprocessing.image.load_img(I_path, target_size=tuple(X_shape[:2]))

    # Init Guided Backprop Model
    model_gb = tf.keras.models.Model(inputs = [MODEL.inputs], outputs = [MODEL.get_layer(name="block_4_conv").output])
    act_layers = [layer for layer in model_gb.layers[1:] if hasattr(layer, "activation")]
    for l in act_layers:
        # Change the ReLU activation to remove the negative gradients
        if l.activation == tf.keras.activations.relu:
            l.activation = ReLu_GuidedBackProp

    # Calculate Guided Backprop
    conv_output_shape = MODEL.get_layer(name="block_4_conv").output.shape[1:]
    plt.figure(figsize=(30, 60))
    # Plot
    plt.subplot(math.ceil((Y_shape+1)/nCols), nCols, 1)
    plt.title("Image")
    plt.imshow(I)
    for i in range(Y_shape):
        # Get random neuron
        n_x = np.random.randint(0, conv_output_shape[0])
        n_y = np.random.randint(0, conv_output_shape[1])
        n_z = np.random.randint(0, conv_output_shape[2])
        # Get outputs of the one neuron in conv layer
        mask = np.zeros((1, *conv_output_shape), dtype=float)
        mask[0, n_x, n_y, n_z] = 1
        # Get grads
        with tf.GradientTape() as tape:
            inputs = tf.cast(np.array([np.array(I)]), tf.float32)
            tape.watch(inputs)
            outputs = model_gb(inputs) * mask
        # Visualise grads
        grads_vis = tape.gradient(outputs, inputs)[0]
        # Visualize outputs of guided backprop
        I_gb = np.dstack((grads_vis[:, :, 0], grads_vis[:, :, 1], grads_vis[:, :, 2]))
        # Normalise to 0 to 1
        I_gb = I_gb - np.min(I_gb)
        I_gb /= I_gb.max()
        # Plot
        plt.subplot(math.ceil((Y_shape+1)/nCols), nCols, i+2)
        plt.title("Class:" + str(i))
        plt.imshow(I_gb)
    plt.show()

# Run
# Params

# Params

# Run