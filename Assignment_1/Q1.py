'''
Question 1

Download the fashion-MNIST dataset and plot 1 sample image for each class as shown in the grid below. 
Use "from keras.datasets import fashion_mnist" for getting the fashion mnist dataset.
'''

# Imports
import json
import wandb
import matplotlib.pyplot as plt

from Dataset import *

# Main Functions
def PlotFashionDataset_ClassWise(X_classwise, img_index=0, num_classes=10, nCols=5):
    plt.figure(figsize=(10, 10))
    nRows = math.ceil(num_classes / nCols)
    for i in range(num_classes):
        plt.subplot(nRows, nCols, i+1)
        plt.imshow(X_classwise[i][img_index], cmap="gray")
        plt.title(DATASET_FASHION_LABELS[i])
    plt.show()

def Wandb_UploadImages(Is, titles):
    Is = np.array(Is)
    for I, title in zip(Is, titles):
        img = wandb.Image(I, caption=title)
        wandb.log({"Q1_grid": img})

# Run
# Params
PLOT_IMG_INDEX = 0
PLOT_NCOLS = 5

WANDB_SAVE = True
# Params

# Run
# Load Data
X_train, X_test, y_train, y_test = LoadFashionDataset()
# Split Data
X_train_classwise = SplitDataset_ClassWise(X_train, y_train)
X_test_classwise = SplitDataset_ClassWise(X_test, y_test)

# Print
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print("X_train_classwise:", [len(X_train_classwise[i]) for i in range(DATASET_N_CLASSES)])
print("X_test_classwise:", [len(X_test_classwise[i]) for i in range(DATASET_N_CLASSES)])

# Visualize Data
PlotFashionDataset_ClassWise(X_train_classwise, PLOT_IMG_INDEX, DATASET_N_CLASSES, PLOT_NCOLS)
PlotFashionDataset_ClassWise(X_test_classwise, PLOT_IMG_INDEX, DATASET_N_CLASSES, PLOT_NCOLS)

# Add to wandb
# Get Wandb Data and Init
WANDB_DATA = json.load(open("config.json", "r"))
WANDB_DATA["use_wandb"] = WANDB_SAVE
if WANDB_DATA["use_wandb"]:
    wandb.init(project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"], name="Q1")
    Is = []
    titles = []
    for i in range(DATASET_N_CLASSES):
        Is.append(X_train_classwise[i][PLOT_IMG_INDEX])
        titles.append(DATASET_FASHION_LABELS[i])
    Wandb_UploadImages(Is, titles)