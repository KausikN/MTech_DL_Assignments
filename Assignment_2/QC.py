'''
Questions Part C

YOLO Weights: https://pjreddie.com/media/files/yolov3.weights
'''

# Imports
import argparse
import functools
from Library.YOLOV3 import *
from Library.VideoUtils import *

# Main Functions
def YOLO_LoadModel(weights_path="Models/yolov3.weights", model_save_path="Models/yolov3.h5"):
    '''
    Load YOLO V3 Model
    '''
    # Create YOLO Model
    yolov3 = YOLO_BuildModel()
    # Load and Set Weights
    wtReader = WeightReader(weights_path)
    wtReader.load_weights(yolov3)
    # Save Yolo Model
    yolov3.save(model_save_path)

    return yolov3

def YOLO_CheckInteraction(v_boxes, v_labels, threshold=0.125):
    '''
    Check for Interactions in the Bounding Boxes
    '''
    interactions = []
    for i in range(len(v_boxes)):
        for j in range(i+1, len(v_boxes)):
            if v_labels[i] == "human" and v_labels[j] == "animal" or v_labels[i] == "animal" and v_labels[j] == "human":
                distance = BBox_Distance(v_boxes[i], v_boxes[j])
                if distance < threshold:
                    interactions.append((i, j))
    return interactions

def YOLO_DetectImage(I, model, interaction_threshold=0.125, model_input_shape=(416, 416)):
    '''
    Detect Interactions and return the final drawn image
    '''
    input_w, input_h = model_input_shape
    # Fix Input Image
    X = cv2.resize(I, (input_h, input_w))
    X = np.array(X, dtype=float) / 255.0
    X, image_w, image_h = expand_dims(X, 0), I.shape[1], I.shape[0]

    # Predict
    yhat = model.predict(X)
    boxes = []
    for i in range(len(yhat)):
        # Decode output of YOLO
        boxes += YOLO_DecodeNetOut(yhat[i][0], YOLO_ANCHORS[i], YOLO_CLASS_THRESHOLD, input_h, input_w)
    # Correct bbox coordinates
    BBox_CorrectScale(boxes, image_h, image_w, input_h, input_w)
    # Non-maximal suppression
    BBox_NMS(boxes, 0.5)
    # Get Bounding Boxes and overall labels
    labels_required = []
    for k in REQUIRED_LABELS.keys(): labels_required.extend(REQUIRED_LABELS[k])
    v_boxes, v_labels, v_scores = BBox_GetLabels(boxes, YOLO_LABELS, list(set(labels_required)), YOLO_CLASS_THRESHOLD)
    v_labels_overall = []
    for l in v_labels:
        for k in REQUIRED_LABELS.keys():
            if l in REQUIRED_LABELS[k]:
                v_labels_overall.append(k)
                break
    v_labels = v_labels_overall

    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    # Check Interactions
    interactions = YOLO_CheckInteraction(v_boxes, v_labels, interaction_threshold)
    interacting_indices = []
    for i, j in interactions: interacting_indices.extend([i, j])
    interacting_indices = list(set(interacting_indices))

    # Draw Bounding Boxes
    I_bbox = BBox_Draw(I, v_boxes, v_labels, v_scores, interacting_indices)

    return I_bbox

# Runner Functions
def Runner_ParseArgs():
    '''
    Parse Args
    '''
    parser = argparse.ArgumentParser(description="Testing for DL Assignment 2 Part C")

    parser.add_argument("--mode", "-m", type=str, default="image", help="'image' or 'video'")
    parser.add_argument("--model", "-ml", type=str, default="Models/PartC_YOLO.h5", help="YOLO model path to use")
    parser.add_argument("--input", "-i", type=str, default="", help="Input file path")
    parser.add_argument("--output", "-o", type=str, default="", help="Output save path")

    parser.add_argument("--threshold", "-th", type=float, default=0.125, help="Interaction Threshold")

    parser.add_argument("--max_frames", "-mf", type=int, default=-1, help="Maximum Frames to Process (-1 for all frames)")
    parser.add_argument("--speedup", "-sp", type=int, default=1, help="Skip Count of frames")

    args = parser.parse_args()
    return args

def Runner_PartC_Test_Image(args):
    '''
    Test YOLO Interaction for Image
    '''
    INTERACTION_THRESHOLD = args.threshold
    # Load YOLO
    # MODEL_yolov3 = YOLO_LoadModel(weights_path, model_save_path)
    MODEL_yolov3 = load_model(args.model)
    # Single Image
    # Load Image
    I = cv2.imread(args.input)
    # Run Model
    I_bbox = YOLO_DetectImage(I, model=MODEL_yolov3, interaction_threshold=INTERACTION_THRESHOLD)
    cv2.imwrite(args.output, I_bbox)

def Runner_PartC_Test_Video(args):
    '''
    Test YOLO Interaction for Image
    '''
    INTERACTION_THRESHOLD = args.threshold
    # Load YOLO
    # MODEL_yolov3 = YOLO_LoadModel(weights_path, model_save_path)
    MODEL_yolov3 = load_model(args.model)
    # Video
    # Init Function
    YOLOFunc = functools.partial(YOLO_DetectImage, model=MODEL_yolov3, interaction_threshold=INTERACTION_THRESHOLD)
    # Run Model
    VideoEffect(args.input, args.output, YOLOFunc, max_frames=args.max_frames, speedUp=args.speedup, fps=20.0)

# Main Vars
weights_path = "Models/yolov3.weights"

# Run
if __name__ == "__main__":
    ARGS = Runner_ParseArgs()
    if ARGS.mode == "image":
        if ARGS.input == "": ARGS.input = "Outputs/TestImg.jpg"
        if ARGS.output == "": ARGS.output = "Outputs/TestImg_Out.jpg"
        Runner_PartC_Test_Image(ARGS)
    elif ARGS.mode == "video":
        if ARGS.input == "": ARGS.input = "Outputs/TestVid.mp4"
        if ARGS.output == "": ARGS.output = "Outputs/TestVid_Out.mp4"
        Runner_PartC_Test_Video(ARGS)
    else:
        print("Invalid Mode!")