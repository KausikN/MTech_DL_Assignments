'''
Predict for all classes and plot
'''

# Imports
import cv2
import matplotlib.pyplot as plt

from Model import *

# Main Functions
def PredictTest_Display(model, n=3, X_shape=(227, 227, 3)):
    # Get N random images from test set for each class
    test_images = {}
    for c in DATASET_INATURALIST_CLASSES:
        Is = []
        for i in range(n):
            Is.append(GetTestImagePath_Random(c))
        test_images[c] = Is
    # Predict Class for each image
    plt.axis("off")
    for c in DATASET_INATURALIST_CLASSES:
        print("True Class:", c)
        for i in range(n):
            I = cv2.imread(test_images[c][i])
            I = cv2.resize(I, tuple(X_shape[:2]))
            I = np.expand_dims(I, axis=0)
            y_pred = model.predict(I)
            y_class = DATASET_INATURALIST_CLASSES[np.argmax(y_pred, axis=-1)[0]]

            plt.subplot(1, n, i+1)
            plt.imshow(cv2.cvtColor(I[0], cv2.COLOR_BGR2RGB))
            plt.title("Predicted: " + y_class)
        plt.show()

# Run
MODEL = Model_LoadModel("Models/Model_PartA.h5")
PredictTest_Display(MODEL, n=3)