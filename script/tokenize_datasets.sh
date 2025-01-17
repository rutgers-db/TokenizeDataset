#!/bin/bash -l

#SBATCH --output=./log.txt
#SBATCH --mem=100g
#!/bin/bash

# Define the dataset_name and feature_name
DATASET_NAME="ArmelR/the-pile-splitted"
FEATURE_NAME="text"

# Define the list of subset_names
SUBSET_NAMES=( "StackExchange") #"ArXiv" "Wikipedia (en)" "OpenSubtitles" "Gutenberg (PG-19)" "YoutubeSubtitles"

# Iterate over each subset_name and run the Python script
for SUBSET_NAME in "${SUBSET_NAMES[@]}"; do
    echo "Processing $SUBSET_NAME..."
    python tokenize_dataset.py --dataset_name "${DATASET_NAME}" --feature_name "${FEATURE_NAME}" --subset_name "${SUBSET_NAME}"
done

echo "All subsets processed.