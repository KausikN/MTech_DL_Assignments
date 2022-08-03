'''
Guided Backpropogation
'''

# Imports
import matplotlib.pyplot as plt
from Model import *

# Main Functions
# Show model filters by plotting @ N Kausik CS21M037
def VisualiseFilter_Display(MODEL, rgb=True, nCols=4):
    '''
    Visualise Filters in Model
    '''
    # Get Second Hidden Layer Parameters
    # filters, biases = MODEL.layers[0].get_weights()
    filters, biases = MODEL.get_layer(name="block_0_conv").get_weights()
    # Normalise to 0-1
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # Show Filters
    plt.figure(figsize=(6, 48))
    N_FILTERS, f_i = filters.shape[3], 1
    for i in range(N_FILTERS):
        f = filters[:, :, :, i]
        if rgb:
            plt.subplot(math.ceil(N_FILTERS/nCols), nCols, f_i)
            plt.axis("off")
            plt.imshow(f)
            f_i += 1
        else:
            for j in range(f.shape[2]):
                plt.subplot(N_FILTERS, f.shape[2], f_i)
                plt.axis("off")
                plt.imshow(f[:, :, j], cmap='gray')
                f_i += 1
    plt.show()

# Run
# Params

# Params

# Run