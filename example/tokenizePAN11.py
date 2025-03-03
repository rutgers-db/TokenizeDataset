"""
This script reads all the text files in the source-document directory of the PAN11 external-detection-corpus,
tokenizes them using the GPT-2 tokenizer, and writes the tokenized data into a single binary file.
"""

import os
import re
from transformers import GPT2TokenizerFast
import numpy as np

# Initialize the GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def get_file_id(filename):
    """
    Extract the numerical part from the filename.
    """
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else None

def tokenize_file(filepath):
    """
    Open the file and read its contents, then tokenize the content.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        tokens_lists = tokenizer(content)['input_ids']
        return tokens_lists

# Define Global Variables
root_dir = "/path/to/pan-plagiarism-corpus-2011"
pan11_dir = os.path.join(root_dir, "external-detection-corpus/")
dataset_dir = os.path.join(pan11_dir, "source-document")
tokenizedFilePath = "PAN11_external_source_gpt2.bin"  # Output binary file path
tokenizedFile = open(tokenizedFilePath, 'wb')  # Open as a binary file

content2write = []

# Iterate through each part directory and tokenize the documents
# The external detection corpus consists of 23 parts
for i in range(1, 24):
    part_dir = os.path.join(dataset_dir, f'part{i}')

    # List all txt files in the directory
    txt_files = [f for f in os.listdir(part_dir) if f.endswith('.txt') and f.startswith('source-document')]
    # Sort files based on their numerical suffix
    txt_files.sort(key=get_file_id)

    # Process each file
    for file in txt_files:
        file_path = os.path.join(part_dir, file)
        tokens = tokenize_file(file_path)
        content2write.append(int(len(tokens)))
        content2write.extend(tokens)
    print(f"Part{i} Completed")

# Save the tokenized data to the binary file
print("Tokenization Completed")
tokenizedFile.write(np.array(content2write, dtype=np.uint32).tobytes())
print("Writing File Completed")