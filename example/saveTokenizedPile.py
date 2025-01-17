# This script loads a tokenized dataset from disk, iterates through it, and writes the tokenized sequences to a binary file.
# The dataset is expected to be tokenized using GPT-2 tokenizer and saved in a specific format.
# The script processes the dataset in chunks to avoid memory overflow and writes the tokenized sequences to a binary file.

import datasets
from datasets import load_from_disk
import numpy as np

# Load the tokenized dataset from the specified directory
tokenized_dataset = load_from_disk("./tokenized_dataset/tokenized_pile")

# Create an iterator for the tokenized dataset
it = iter(tokenized_dataset)

# Specify the output binary file to store the encoded sequences
encodedFile = "./pile_gpt2_test.bin"
encodedSeqF = open(encodedFile, 'wb')

# Initialize a buffer to store the tokenized sequences
buffer = []

# Iterate through the dataset and process each row
for i in range(0, tokenized_dataset.num_rows):
    # Get the 'input_ids' from the current row
    tokens = next(it)['input_ids']
    
    # Append the length of the token sequence to the buffer
    buffer.append(len(tokens))
    
    # Append the token sequence to the buffer
    buffer.extend(tokens)
    
    # If the buffer exceeds 100 million elements, write it to the binary file
    if len(buffer) > 100000000:
        encodedSeqF.write(np.array(buffer, dtype=np.uint32).tobytes())
        # Clear the buffer after writing to the file
        buffer = []
        # Print the progress
        print(str(i) + "/" + str(tokenized_dataset.num_rows))

# Write any remaining tokens in the buffer to the binary file
encodedSeqF.write(np.array(buffer, dtype=np.uint32).tobytes())

# Close the binary file
encodedSeqF.close()
