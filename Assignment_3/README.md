# cs6910_assignment3
CS6910 Deep Learning Assignment 3 Code

By,

Karthikeyan S (CS21M028)

N Kausik (CS21M037)

# Dakshina Tamil Transliteration
Command to run the code:
```shell
python main.py 
    --mode {mode} 
    --model {model}
    --dataset {dataset}
    --epochs {epochs}
    --batch_size {batch_size}
    --encoder {encoder}
    --encoder_embedding_size {encoder_embedding_size}
    --encoder_n_units {encoder_n_units}
    --act_func {act_func}
    --dropout {dropout}
    --attention
```

Parameters are,

    - mode: "train" or "test" or "vh" (Visulaise Attention Heatmap) or "vc" (Visualise Connectivity)
    - model:
        - save path of model for training
        - load path of model for testing, attention heatmap, visualising connectivity
    - dataset: dataset path
        - should contain the files /ta/lexicons/ta.translit.sampled.{x}.tsv with {x} as "train", "test" and "dev"
    - epochs: number of epochs
    - batch_size: batch size
    - encoder: Encoder and Decoder type (same type is used)
        - "RNN" or "GRU" or "LSTM"
    - encoder_embedding_size: Embedding size for encoder and decoder (same size is used)
    - encoder_n_units: Number of units for encoder and decoder (same number is used)
    - act_func: Activation function for encoder and decoder (same function is used)
    - dropout: Dropout and Recurrent Dropout rate for encoder and decoder (same dropout is used)
    - attention: If given, attention is used

# GPT Lyrics Generator
Command to run the code:
```shell
python LyricGenerator.py 
    --mode {mode} 
    --model {model}
    --dataset {dataset}
    --epochs {epochs}
    --batch_size {batch_size}
    --prompt {prompt}
    --songs {songs}
```

Parameters are,

    - mode: "train" or "gen" (Generate Lyrics)
    - model:
        - save path of model for training
        - load path of model for generating
    - dataset: dataset path
        - should contain the merged txt file with all lyrics
    - epochs: number of epochs
    - batch_size: batch size
    - prompt: Prompt to be used for generating lyrics
    - songs: Number of songs to be generated