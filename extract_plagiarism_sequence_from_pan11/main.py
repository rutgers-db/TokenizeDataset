import os
import numpy as np
from utils import extract_plagiarism_features, extract_sequence_from_file, extract_txt_filenames, replace_str_suffix, jaccard_similarity, get_file_content
from transformers import GPT2TokenizerFast

# Define the similarity threshold
# If you want to filter out the pairs with low similarity, you can set this threshold
# sim_thres = 0.8

# Initialize tokenizer
# You can choose your own tokenizer here
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Define paths
# The path to your pan11 dataset directory
pan11_root_dir = "/to/your/path/pan-plagiarism-corpus-2011"  # Update with your path
suspicious_root_dir = os.path.join(pan11_root_dir, "external-detection-corpus/suspicious-document")
source_dir = os.path.join(pan11_root_dir, "external-detection-corpus/source-document")


def tokenize_str(str):
    return tokenizer(str)['input_ids']

def tokenize_file(filepath):
    """
    Tokenize the contents of a file.

    Args:
    filepath (str): Path to the file.

    Returns:
    list: List of token IDs.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        tokens_lists = tokenizer(content)['input_ids']
        return tokens_lists


# Containers for data
suspicious_sequences = []
source_documents_set = set()
source_sus_pairs = {}

# Extract filenames and their paths from the source directory
source_files_dict = extract_txt_filenames(source_dir)

# Process each XML file in the suspicious documents directory
for subdir in next(os.walk(suspicious_root_dir))[1]:
    print("Walking ",subdir)
    subdir_path = os.path.join(suspicious_root_dir, subdir)
    for file in os.listdir(subdir_path):  # List files in each subdirectory
        if file.endswith('.xml'):
            xml_path = os.path.join(subdir_path, file)
            features = extract_plagiarism_features(xml_path)

            if len(features) == 0:
                continue

            # Change XML file suffix to get corresponding TXT file path
            txt_file_path = replace_str_suffix(xml_path)
            # Process each plagiarism feature
            for feature in features:
                this_offset = int(feature['this_offset'])
                this_length = int(feature['this_length']) 
                source_offset = int(feature['source_offset'])
                source_length = int(feature['source_length'])
                source_file = feature['source_reference']

                # Get the suspicious sequence 
                sus_sequence = extract_sequence_from_file(txt_file_path, this_offset, this_length)
                if(len(sus_sequence) == 0):
                    continue
                
                # Get the source Squence and verify their similarity
                source_path = source_files_dict[source_file]
                source_sequence = extract_sequence_from_file(source_path, source_offset, source_length)
                tokens_1 = tokenize_str(sus_sequence)
                tokens_2 = tokenize_str(source_sequence)
                sim = jaccard_similarity(tokens_1, tokens_2)
                
                # if sim > sim_thres:
                suspicious_sequences.append(sus_sequence)
                # Add the source document to the set
                source_documents_set.add(feature['source_reference'])
                if source_file not in source_sus_pairs:
                    source_sus_pairs[source_file] = []
                source_sus_pairs[source_file].append(len(suspicious_sequences) - 1)


# Debug: Show the first source suspicious pair:
# for source_file in source_documents_set:
#     # Show the suspicious sequences
#     print("Showing the suspicious file")
#     for sus_id in source_sus_pairs[source_file]:
#         print(suspicious_sequences[sus_id])
#         print("----------------------------")
    
#     print("Showing the source")
#     source_path = source_files_dict[source_file]
#     print(get_file_content(source_path))
#     print(source_file)
#     break

# Tokenize suspicious sequences and prepare them for binary writing
tokenized_suspicious_binary = []
for sequence in suspicious_sequences:
    tokens = tokenizer(sequence)['input_ids'] 
    tokenized_suspicious_binary.append(int(len(tokens)))
    tokenized_suspicious_binary.extend(tokens)

# Tokenize source documents and prepare them for binary writing
tokenized_sources_binary = []
for source_file in source_documents_set:
    source_path = source_files_dict[source_file]
    tokens = tokenize_file(source_path)
    tokenized_sources_binary.append(int(len(tokens)))
    tokenized_sources_binary.extend(tokens)

# Mkdir the data directory if it does not exist
if not os.path.exists('./data'):
    os.mkdir('./data')

# Write the similar pairs to txt files
with open('./data/all_source_sus_pairs.txt', 'w') as file:
    # Iterate over the dictionary items
    for source_file in source_documents_set:
        # Convert the list of integers to a string
        value_list = source_sus_pairs[source_file]
        value_str = ', '.join(map(str, value_list))
        # Write the key and value string to the file
        file.write(f"{value_str}\n")

# Write to binary files
with open('./data/all_suspicious_seqs_pan11.bin', 'wb') as f:
    f.write(np.array(tokenized_suspicious_binary, dtype=np.uint32).tobytes())

with open('./data/all_source_with_plagiarism.bin', 'wb') as f:
    f.write(np.array(tokenized_sources_binary, dtype=np.uint32).tobytes())

print("Data processing and writing completed.")
print("Suspicious Sequences Num:", len(suspicious_sequences))
print("Corresponding Source Documents Num:", len(source_documents_set))