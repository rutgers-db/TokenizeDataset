'''
This script is to tokenize downloaded dataset
So before running this script, you should run download_dataset.py first to obtain a downloaded dataset
'''
import os
import numpy as np
from langdetect import detect, LangDetectException
from transformers import GPT2TokenizerFast
import re
import argparse
import datasets
from datasets import load_dataset
from pathlib import Path

target_path = "./downloaded_dataset"  # the location where the dataset saves
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_path)

# Before running this script, you should export your environment variable if you have reset the environment variable
# HF_CACHE = "/research/projects/zp128/dataset_tokenizedGbt2/downloaded_dataset/hf_datasets_cache"
# HF_CACHE = "/path/to/hf_datasets_cache"

# Indicate dataset configuration

# openwebtext
# dataset_name = "openwebtext"
# feature_name = "text"

# dataset_name = "c4"
# split_name = "train"
# subset_name = "en"
# feature_name = "text"

# CNN Daily Mail
# dataset_name = "cnn_dailymail"
# subset_name = "3.0.0"
# split_name = "test"
# feature_name = "article"

# SST2
# dataset_name = "sst2"
# split_name = "train"
# feature_name = "sentence"

# squad
# dataset_name = "squad"
# split_name = "validation"
# feature_name = "context"

# pile
# dataset_name = "ArmelR/the-pile-splitted"
# feature_name = "text"
# subset_name = "Pile-CC"

parser = argparse.ArgumentParser(description='Tokenize a dataset.')
# Add the arguments
parser.add_argument('--dataset_name', type=str, required=True,
                    help='The name of the dataset to process')
parser.add_argument('--feature_name', type=str, required=True,
                    help='The feature of the dataset to tokenize')
parser.add_argument('--split_name', type=str, default=None,
                    help='The split name of the dataset to tokenize')
parser.add_argument('--subset_name', type=str, default=None,
                    help='The subset of the dataset to process (optional)')

# Parse the arguments
args = parser.parse_args()
dataset_name = args.dataset_name
feature_name = args.feature_name
subset_name = args.subset_name
split_name = args.split_name

# Load(Or Download) dataset
print("Loading Dataset %s" % dataset_name)

# Ensure variables are defined or set them to None if they're not
subset_name = locals().get('subset_name', None)
split_name = locals().get('split_name', None)
if subset_name and split_name:
    # Both subset_name and split_name are defined
    dataset = load_dataset(dataset_name, subset_name,
                           split=split_name, cache_dir=HF_CACHE)
elif subset_name and not split_name:
    # Only subset_name is defined
    dataset = load_dataset(dataset_name, subset_name, cache_dir=HF_CACHE)
elif not subset_name and split_name:
    # Only split_name is defined
    dataset = load_dataset(dataset_name, split=split_name, cache_dir=HF_CACHE)
else:
    # Neither subset_name nor split_name is defined
    dataset = load_dataset(dataset_name, cache_dir=HF_CACHE)


def sanitize_filename(filename):
    """
    Sanitizes a string to make it a legal file name.
    Replaces spaces, parentheses, and other potentially illegal characters with underscores.
    """
    # Define a regular expression pattern for characters you want to replace with "_"
    # This pattern targets spaces, parentheses, and could be extended to include other characters
    pattern = r'[ \(\)]'

    # Replace the targeted characters with "_"
    sanitized = re.sub(pattern, '_', filename)

    return sanitized


# Indicate tokenized_file configuration
dataset_suffix = dataset_name.split("/")[-1]
# GPT4 tokenizer
tokenizer_name = "gpt4"
tokenizer_para = "Xenova/gpt-4"

# GPT2 tokenizer
# tokenizer_name = "gpt2"
# tokenizer_para = "gpt2"

tokenizedFilePath = f"./tokenized_bin/{dataset_suffix}_{
    subset_name + '_' if subset_name else ''}{tokenizer_name}.bin"
tokenizedFilePath = sanitize_filename(tokenizedFilePath)
tokenizedFile = open(tokenizedFilePath, 'wb')  # open as a binary file

tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_para)


def is_english(text, default="en"):
    """
    Determines if the given text is in English.

    This function attempts to detect the language of the provided text. If the detection fails,
    it assumes the text is in the default language, which is English by default.

    Args:
        text (str): The text to be analyzed for language detection.
        default (str, optional): The default language to assume in case of detection error. Defaults to "en".

    Returns:
        bool: True if the detected language is English, False otherwise.
    """
    try:
        # Detect the language of the text
        language = detect(text)
    except LangDetectException:
        # In case of detection error, assume default language
        language = default
    # Return True if the detected language is English, False otherwise
    return language == "en"


# Initialize global counters
total_sentences = 0
non_english_sentences = 0

# Map function used in dataset.map().
# Used in batch mode for default dataset
def tokenize_and_save_binary(batched_items):
    global total_sentences, non_english_sentences
    content2write = []

    # Increment total sentences by the batch size
    total_sentences += len(batched_items[feature_name])

    # Filter non-English Sentences for batched_items[feature_name] and count them
    english_sentences = []
    for sentence in batched_items[feature_name]:
        if is_english(sentence):
            english_sentences.append(sentence)
        else:
            non_english_sentences += 1

    # tokenize
    tokens_lists = tokenizer(english_sentences)['input_ids']

    # merge these tokens with each length
    for tokens in tokens_lists:
        content2write.append(int(len(tokens)))
        content2write.extend(tokens)

    # Write the tokenized data to the data
    tokenizedFile.write(np.array(content2write, dtype=np.uint32).tobytes())

    # if running in ilab (our school's cluster), pls keep job so that it won't be killed
    os.system("keep-job 48")


print("Tokenizing and Saving Data")
dataset.map(tokenize_and_save_binary, batched=True)
print("Tokenization Completed")

# Calculate and print the percentage of non-English sentences
if total_sentences > 0:
    percentage_non_english = (non_english_sentences / total_sentences) * 100
    print(f"Tokenization Completed. Filtered out {non_english_sentences} non-English sentences out of {
        total_sentences} total sentences ({percentage_non_english:.2f}%).")
else:
    print("No sentences were processed.")

# Below is for squad, a specific dataset that need below process to deduplicate the entries if you want
# total_entries = 0  # Counter for total entries
# unique_entries = 0  # Counter for unique entries
# # Tokenize and save, avoiding duplicate contexts
# with open(tokenizedFilePath, 'wb') as tokenizedFile:
#     previous_context = None  # Initialize previous context

#     for item in dataset:
#         current_context = item[feature_name]
#         total_entries += 1
#         # Check if the current context is the same as the previous one
#         if current_context != previous_context:
#             unique_entries += 1  # Increment unique entries counter
#             # Tokenize the current context
#             tokens = tokenizer(current_context)['input_ids']
#             content2write = [int(len(tokens))] + tokens
#             tokenizedFile.write(np.array(content2write, dtype=np.uint32).tobytes())

#             # Update the previous context
#             previous_context = current_context

#         # Custom command for specific environments (optional)
#         os.system("keep-job 48")

# print(f"Tokenization completed. {unique_entries} unique contexts out of {total_entries} total entries.")
