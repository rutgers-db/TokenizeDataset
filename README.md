# Hugging Face Dataset Downloader and Tokenizer

This repository contains Python scripts for downloading and tokenizing datasets from the Hugging Face [Datasets](https://huggingface.co/datasets) library.

## Requirements

To use these scripts, you will need:

- Python 3.x
- The Hugging Face [datasets](https://huggingface.co/docs/datasets/) library (installable via pip)
- A Conda environment (recommended, but not required)
- An environment variable `HF_DATASETS_CACHE` to indicate where the cache stores so that it won't exceed the space of the default disk.

To install the `datasets` library via pip, run:

    pip install datasets



Before using the scripts, you should set the `HF_DATASETS_CACHE` environment variable to indicate where the cache will be stored. For example, to set it to a directory called `hf_datasets_cache` in your specified directory, run:

    export HF_DATASETS_CACHE="$YOURDIR/hf_datasets_cache"



## Usage

To download a dataset, run the `download_dataset.py` script. By default, it downloads the `c4` dataset to a subdirectory in the `downloaded_dataset/hf_datasets_cache` directory.

    python download_dataset.py


To tokenize a downloaded dataset, run the `tokenize_dataset.py` script with the desired tokenization method as an argument. By default, it tokenizes the `c4` dataset using the GPT-2 tokenizer from the Hugging Face `transformers` library, and saves the resulting tokenized dataset to a binary file called `c4_en_train_gpt2` in the `tokenized_bin` directory.

    python tokenize_dataset.py


Note that you can specify a different dataset by modifying the `dataset_name` , `split_name`, `subset_name` variables in the `download_dataset.py` script, and you can specify a different tokenization method by modifying variable by yourself in the `tokenize_dataset.py` script.

## Example
To check some example usages(i.e. download and tokenize openwebtext), please check the codes in the folder `example`
