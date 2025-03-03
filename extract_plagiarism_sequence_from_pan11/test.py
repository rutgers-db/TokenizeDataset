import utils
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2TokenizerFast
import sys

# Initialize the GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Test string for tokenization
test_str = "Hello, world!"
print(tokenizer.encode(test_str))
print(tokenizer(test_str))

# Exit the script before running the rest of the code
sys.exit()

# Extract plagiarism features from the XML file
features = utils.extract_plagiarism_features("/research/projects/zp128/dataset_tokenizedGbt2/pan11/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document/part1/suspicious-document00334.xml")

# Extract suspicious sequence from the text file
offset_len = int(features[0]['this_length'])
offset = int(features[0]['this_offset'])
sus_seq = utils.extract_sequence_from_file("/research/projects/zp128/dataset_tokenizedGbt2/pan11/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document/part1/suspicious-document00334.txt", offset, offset_len)
print(sus_seq)

# Extract source sequence from the text file
offset_len = int(features[0]['source_length'])
offset = int(features[0]['source_offset'])
source_txt = features[0]['source_reference']

root_path = "/research/projects/zp128/dataset_tokenizedGbt2/pan11/pan-plagiarism-corpus-2011/external-detection-corpus/source-document"
source_files_dict = utils.extract_txt_filenames(root_path)
source_file_name = features[0]['source_reference']
source_seq = utils.extract_sequence_from_file(source_files_dict[source_file_name], offset, offset_len)
print(source_seq)

# Initialize sentence transformers models
model = SentenceTransformer('all-MiniLM-L6-v2')
model2 = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# List of sentences for embedding
sentences = [
    "This movie is very exciting",
    "This is a very good movie",
    "The weather is really nice today"
]

# Compute embeddings
embeddings = model.encode(sentences)
embeddings2 = model2.encode(sentences)

# Compute similarity between the first two sentences
similarity = util.cos_sim(embeddings[0], embeddings[1])
similarity2 = util.cos_sim(embeddings2[0], embeddings[1])

print(f"Similarity between the first two sentences: {similarity.item():.4f}")  # Should be high
print(f"Similarity between the first two sentences (model2): {similarity2.item():.4f}")  # Should be high

# Compute similarity between the first and third sentences
similarity = util.cos_sim(embeddings[0], embeddings[2])
print(f"Similarity between the first and third sentences: {similarity.item():.4f}")  # Should be low
similarity2 = util.cos_sim(embeddings2[0], embeddings[2])
print(f"Similarity between the first and third sentences (model2): {similarity2.item():.4f}")  # Should be low