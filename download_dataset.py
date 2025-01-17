'''
This script is to download a dataset from huggingface
Note that to use enough space to store cache, you should enter export HF_DATASETS_CACHE="/research/projects/zp128/dataset_tokenizedGbt2/downloaded_dataset/hf_datasets_cache" in the command window
That means you indicate a specified directory to store cache temporarily, otherwise the default place is ~/.cache which highly probably bursts the disk
'''
import datasets
from datasets import load_dataset
from pathlib import Path

target_path = "./downloaded_dataset"  # the location where the dataset saves
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_path)

# Example dataset configuration
# Indicate dataset configuration
# dataset_name = "c4"
# split_name = "train"
# subset_name = "en"

# dataset_name = "wiki40b"
# split_name = "train"
# subset_name = "en"

# Download dataset
print("Start Downloading")
dataset = load_dataset(dataset_name, subset_name, split=split_name)
print("%s of %s  is downloaded" % (split_name, dataset_name))
