# cs6910_assignment2
CS6910 Deep Learning Assignment 2 Code

By,

Karthikeyan S (CS21M028)

N Kausik (CS21M037)

# Dataset Setup
- Run 
```shell
python SplitDataset.py 
    --dataset {dataset}
```
- Here, {dataset} is the path to the dataset folder (must have "train" folder inside it)
- This script generates the "validation" folder inside the same dataset folder by moving images from "train" to "validation"
- Running this script is done ONLY once for a dataset folder

# Part A
Command to run the code:
```shell
python QA.py 
    --mode {mode} 
    --model {model}
    --dataset {dataset}
    --epochs {epochs}
    --batch_size {batch_size}
    --filter_size {filter_size}
    --n_filters {n_filters}
    --dropout {dropout}
    --no_batch_norm
    --dense_neurons {dense_neurons}
    --dense_dropout {dense_dropout}
    --learning_rate {learning_rate}
    --n_cols {n_cols}
    --filter_rgb {filter_rgb}
```

Parameters are,

    - mode: "train" or "test" or "gb" (Guided Backprop) or "vf" (Visualise Filters)
    - model:
        - save path of model for training
        - load path of model for testing, guided backprop, visualising filters
    - dataset: dataset path
        - should contain the folders "train", "val" (test) and "validation" (generated in dataset setup stage)
    - epochs: number of epochs
    - batch_size: batch size
    - filter_size: filter sizes for the 5 layers (Eg. "3,4,5,6,7")
    - n_filters: number of filters for each layer (Eg. "32,64,128,256,512")
    - dropout: dropout rate for the conv-relu-maxpool blocks
    - no_batch_norm: if given, batch normalisation is not used
    - dense_neurons: number of neurons to use in the dense layer
    - dense_dropout: dropout rate for the dense layer
    - learning_rate: learning rate for adam optimizer
    - n_cols: number of columns used in visualising filters (only used when filter_rgb is True)
    - filter_rgb: if given, filters are visualised in RGB format instead of separately

# Part B
Command to run the code:
```shell
python QB.py 
    --mode {mode} 
    --model {model}
    --dataset {dataset}
    --epochs {epochs}
    --batch_size {batch_size}
    --no_data_aug
    --unfreeze_count {unfreeze_count}
    --dense_neurons {dense_neurons}
    --dense_dropout {dense_dropout}
    --learning_rate {learning_rate}
```

Parameters are,

    - mode: "train" or "test"
    - model:
        - save path of model for training
        - load path of model for testing
    - dataset: dataset path
        - should contain the folders "train", "val" (test) and "validation" (generated in dataset setup stage)
    - epochs: number of epochs
    - batch_size: batch size
    - no_data_aug: if given, data augmentation is not used
    - unfreeze_count: number of layers to unfreeze in pretrained model
    - dense_neurons: number of neurons to use in the dense layer
    - dense_dropout: dropout rate for the dense layer
    - learning_rate: learning rate for adam optimizer

# Part C
Command to run the code:
```shell
python QC.py 
    --mode {mode} 
    --model {model}
    --input {input}
    --output {output}
    --threshold {threshold}
    --max_frames {max_frames}
    --speedup {speedup}
```

Parameters are,

    - mode: "image" or "video"
    - model: load path of YOLO model
    - input: input file path
        - if mode is image: input image path (Eg. "input.jpg")
        - if mode is video: input video path (Eg. "input.mp4")
    - output: output file path
        - if mode is image: output image path (Eg. "output.jpg")
        - if mode is video: output video path (Eg. "output.mp4")
    - threshold: threshold for checking interactions between bounding boxes of human and animal
        - if threshold * (image_width + image_height) / 2 > distance between bounding boxes, then interaction is detected
    - max_frames: maximum number of frames to process in video (used only when mode is video)
    - speedup: number of frames to skip in each iteration (used only when mode is video)
        - Eg. if speedup is 2, then every other frame is processed

Example Processed Video:

[![Example Processed Video](https://img.youtube.com/vi/4egDPyeDbVI/0.jpg)](https://youtu.be/4egDPyeDbVI)