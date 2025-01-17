import os
import re
from transformers import GPT2TokenizerFast
import numpy as np

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def get_file_id(filename):
    # Extract the numerical part from the filename
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else None

def tokenize_file(filepath):
    # Open the file and read its contents
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        tokens_lists = tokenizer(content)['input_ids']
        return tokens_lists

# Define Global Variables
dataset_name = "PAN11"
pan11_dir = "/research/projects/zp128/dataset_tokenizedGbt2/pan11/pan-plagiarism-corpus-2011/external-detection-corpus/"
dataset_dir = pan11_dir + "source-document"
tokenizedFilePath = "./tokenized_bin/PAN11_external_source_gpt2.bin"  # ./PAN11_external_suspicious.bin
tokenizedFile = open(tokenizedFilePath,'wb') # open as a binary file

content2write = []

# This is a command to keep the job running in my university's cluster
# You can remove it if you are running the code in your local machine
os.system("keep-job 72")

# Iterate each documents and tokenize it 
for i in range(1, 23):
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

# Save the data
print("Tokenization Completed")
tokenizedFile.write(np.array(content2write,dtype=np.uint32).tobytes())
print("Writing File Completed")