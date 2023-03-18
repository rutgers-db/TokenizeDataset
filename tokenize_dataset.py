'''
This script is to tokenize downloaded dataset
So before running this script, you should run download_dataset.py first to obtain a downloaded dataset
'''
import datasets
from datasets import load_dataset
from pathlib import Path

target_path = "./downloaded_dataset" # the location where the dataset saves
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_path)

# Indicate dataset configuration
dataset_name = "c4"
split_name = "train"
subset_name = "en"

# Download dataset
print("Loading Dataset %s" % dataset_name)
dataset = load_dataset(dataset_name, subset_name, split=split_name)

# Indicate tokenized_file configuration
tokenizedFilePath = "./c4_en_train_gpt2.bin"
tokenizedFile = open(tokenizedFilePath,'wb') # open as a binary file

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
import numpy as np
import os

# Map function used in dataset.map().
# Used in batch mode
def tokenize_and_save_binary(batched_items):
    content2write = []
    
    # tokenize 
    tokens_lists = tokenizer(batched_items['text'])['input_ids']

    # merge these tokens with each length
    for tokens in tokens_lists:
        content2write.append(int(len(tokens)))
        content2write.extend(tokens)
    
    # write
    tokenizedFile.write(np.array(content2write,dtype=np.uint32).tobytes())

    # if running in ilab, pls keep job so that it won't be killed
    os.system("keep-job 48")

print("Tokenizing and Saving Data")
dataset.map(tokenize_and_save_binary, batched = True)
print("Tokenization Completed")