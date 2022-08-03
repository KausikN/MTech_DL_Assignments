"""
Visualise Connectivity of Model
"""

# Imports
import os
import cv2
from moviepy.editor import ImageSequenceClip

from Model import *

# Main Functions
def Vis_Connectivity(model, dataset, dataset_encoded, n=1):
    '''
    Visualise Connectivity of Model
    '''
    temp_path = "Outputs/Vis_Connectivity_Temp.png"
    # Get Test Dataset
    DATASET_TEST, DATASET_ENCODED_TEST = dataset, dataset_encoded
    # Get Encoder Decoder Models
    params = {
        "target_chars": DATASET_ENCODED_TEST["chars"]["target_chars"],
        "target_char_map": DATASET_ENCODED_TEST["chars"]["target_char_map"]
    }
    params["use_attention"] = "attention" in [layer.name for layer in model.layers]
    encoder_model, decoder_model = Model_Inference_GetEncoderDecoder(model, **params)
    # Get Encoder Decoder Inputs
    dataset_test_encoder_input = np.argmax(DATASET_ENCODED_TEST["encoder_input"], axis=-1)

    # Get Random Test Samples
    random_indices = [np.random.randint(0, dataset_test_encoder_input.shape[0]) for i in range(n)]
    for i in random_indices:
        # Get Output
        query = dataset_test_encoder_input[i:i+1]
        decoded_word, attention = Model_Inference_Transliterate(query, encoder_model, decoder_model, **params)
        decoded_word = decoded_word[0]
        attention = np.array(attention)
        attention = attention.reshape((attention.shape[0], attention.shape[-1]))
        attention = attention[:, 1:] # Skip Start Symbol for Input word

        # Form GIF
        frames = []
        input_word = DATASET_TEST["input"][i]
        for li in range(len(decoded_word)):
            plt.axis("off")
            # Show Decoded Text
            decoded_alphas = np.zeros(len(decoded_word))
            decoded_alphas[li] = 0.5
            for j in range(len(decoded_word)):
                x_loc = 0.1 + 0.8 * (j / len(decoded_word))
                decoded_tf = plt.text(x_loc, 0.9, decoded_word[j], fontproperties=FontProperties(fname=FONT_PATH_TAMIL), fontsize=30)
                decoded_tf.set_bbox(dict(alpha=decoded_alphas[j], facecolor="blue", edgecolor="blue"))
            # Show Input Text
            input_alphas = attention[li]
            for j in range(len(input_word)):
                x_loc = 0.1 + 0.8 * (j / len(input_word))
                input_tf = plt.text(x_loc, 0.5, input_word[j], fontsize=30)
                input_tf.set_bbox(dict(alpha=input_alphas[j], facecolor="blue", edgecolor="blue"))
            # Save Frame
            plt.savefig(temp_path)
            plt.clf()
            # plt.show()
            frames.append(cv2.cvtColor(cv2.imread(temp_path), cv2.COLOR_BGR2RGB))
        # Create GIF
        connClip = ImageSequenceClip(frames, fps=5)
        connClip.write_gif(f"Outputs/Vis_Connectivity_{i}.gif", fps=5)
        # Delete Temp
        os.remove(temp_path)

# Run