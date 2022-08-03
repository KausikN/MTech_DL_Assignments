"""
Lyric Generator for Question 8
"""

# Imports
import os
import argparse
from sklearn.model_selection import train_test_split
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer

# Main Vars
SCRIPT_PATH = "./GPT2/run_clm.py"
DATASET_PATH = "./GPT2/data.txt"

# Lyric Gen Functions
def LyricGen_PreprocessDataset(save_dir="./GPT2/Outputs/"):
    '''
    Preprocesses the dataset for the Lyric Generator
    '''
    global DATASET_PATH
    with open(DATASET_PATH, "r", encoding="utf-8") as data:
        dataset = ["<|title|>" + x.strip() for x in data.readlines()]
    train, val = train_test_split(dataset, train_size=.75, random_state=2022)

    # Now load the data line by line
    with open(os.path.join(save_dir, "train.txt"), "w", encoding="utf-8") as file_handle:
        file_handle.write("<|endoftext|>".join(train))
    with open(os.path.join(save_dir, "val.txt"), "w", encoding="utf-8") as file_handle:
        file_handle.write("<|endoftext|>".join(val))

def LyricGen_RunScript(n_epochs=1, batch_size=128, save_dir="./GPT2/Outputs/", model_dir="./GPT2/Models/"):
    savepath_train = os.path.join(save_dir, "train.txt")
    savepath_val = os.path.join(save_dir, "val.txt")
    model = "gpt2-medium" # gpt2-medium, distilgpt2
    # Form CMD
    CMD = f"""python {SCRIPT_PATH} \
--model_type {model} \
--model_name_or_path {model} \
--train_file '{savepath_train}' \
--do_train \
--validation_file '{savepath_val}' \
--do_eval \
--per_gpu_train_batch_size {batch_size} \
--save_steps -1 \
--num_train_epochs {n_epochs} \
--fp16 \
--output_dir='{model_dir}'"""
    # Run CMD
    os.system(CMD)

def LyricGen_LoadModel(model_dir="./GPT2/Models/"):
    '''
    Loads the model for the Lyric Generator
    '''
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = TFGPT2LMHeadModel.from_pretrained(model_dir)
    return model, tokenizer

def LyricGen_Generate(model, tokenizer, prompt="I love deep learning", n_songs=1, save_dir="./GPT2/Outputs/"):
    '''
    Generates a lyric
    '''
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    # Generate
    lyrics = model.generate(
        input_ids, 
        max_length=200,  
        num_return_sequences=n_songs,
        no_repeat_ngram_size=4,
        repetition_penalty=2.2,
        top_p=0.92,
        temperature=0.85,
        do_sample=True,
        top_k=128,
        early_stopping=True
    )
    # Display
    for i, song_data in enumerate(lyrics):
        print("{}: {}".format(i, tokenizer.decode(song_data, skip_special_tokens=True)))
        print()
    # Save
    fileLines = []
    for i, song_data in enumerate(lyrics):
        fileLines.append("{}: {}\n".format(i, tokenizer.decode(song_data, skip_special_tokens=True)))
    open(os.path.join(save_dir, "gen_lyrics.txt"), "w", encoding="utf-8").write("\n".join(fileLines))

# Runner Functions
def Runner_ParseArgs():
    '''
    Parse Args
    '''
    global DATASET_PATH

    parser = argparse.ArgumentParser(description="Lyric Generation using GPT2")
    parser.add_argument("--mode", "-m", type=str, default="gen", help="train | gen")
    parser.add_argument("--model", "-ml", type=str, default="./GPT2/Models/", help="Model path to use or save to")
    parser.add_argument("--dataset", "-dt", type=str, default=DATASET_PATH, help="Dataset path to use")
    # Train Args
    parser.add_argument("--epochs", "-e", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    # Gen Args
    parser.add_argument("--prompt", "-p", type=str, default="I love deep learning", help="Prompt to use for generation")
    parser.add_argument("--songs", "-s", type=int, default=1, help="Number of songs to generate")

    args = parser.parse_args()
    DATASET_PATH = args.dataset
    return args

def Runner_Train(args):
    '''
    Runner function for Lyric Generator Training
    '''
    LyricGen_PreprocessDataset()
    LyricGen_RunScript(args.epochs, args.batch_size, model_dir=args.model)

def Runner_Generate(args):
    '''
    Runner function for Lyric Generator Generation
    '''
    model, tokenizer = LyricGen_LoadModel(args.model)
    LyricGen_Generate(model, tokenizer, args.prompt, args.songs)

# Run
if __name__ == "__main__":
    args = Runner_ParseArgs()
    if args.mode == "train":
        Runner_Train(args)
    elif args.mode == "gen":
        Runner_Generate(args)
    else:
        print("Invalid mode")