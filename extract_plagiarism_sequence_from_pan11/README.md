# Suspicious Plagiarism Sequence Extractor for PAN11

## Overview
This repository is dedicated to the task of identifying and extracting suspicious plagiarism sequences from the PAN11 dataset. The PAN11 dataset includes a collection of documents, among which only a subset contains instances of plagiarism. The primary goal of this repository is to isolate these instances for further analysis.

## Features
The repository facilitates the following key operations:

1. **Extraction of Suspicious Sequences**: Identifies and extracts sequences from suspicious documents that potentially contain plagiarism, along with their corresponding source documents from the PAN11 dataset.

2. **Query Dataset Creation**: Converts all extracted suspicious sequences into a query dataset. This dataset is designed for use in plagiarism detection and analysis.

3. **Source Document Compilation**: Aggregates all relevant source documents, which are suspected of being plagiarized, into a single file for ease of access and analysis.

4. **Tokenization**: Processes both the suspicious sequences and the source documents, breaking them down into tokens for detailed textual analysis.

## Output
The processing results in the generation of two primary files:
- `suspicious_seqs_pan11.bin`: Contains all the suspicious sequences extracted from the PAN11 dataset.
- `source_with_plagiarism.bin`: A compiled file of source documents that correspond to the suspicious sequences.

## Usage
   ```bash
   # Activate your conda environment (recommended but optional)
   conda activate your_env

   # Run the main script
   python3 main.py
   ```
## Reminder
This function processes the PAN11 dataset for text classification tasks. It reads the dataset from the specified path,
tokenizes the text using the provided tokenizer, and writes the tokenized text into locals.

Parameters:
- dataset_path (str): The path to the PAN11 dataset file. You need to specify this path in the main Python code.
- tokenizer (Tokenizer): The tokenizer to be used for tokenizing the text. You need to specify this tokenizer in the main Python code.

Note: The main.py script does not accept command-line arguments. You need to manually specify the `dataset_path` and `tokenizer` parameters within the main Python code.
"""

## PAN11 Dataset
The PAN11 dataset can be accessed and downloaded from the following link: [PAN11 Dataset on Zenodo](https://zenodo.org/records/3250095)