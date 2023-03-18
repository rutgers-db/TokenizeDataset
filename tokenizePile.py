import datasets
import os
from datasets import load_from_disk
# from transformers import GPT2TokenizerFast
# # tokenize the train split of pile and then save it in a bin file

# # Download Dataset
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

# # Load Tokenizer and Get Vocab Size
# cnt = 0 
# def tokenize_map_function(items):
#     # iterate each item in a batch
#     # print(type(items["text"]))
#     global cnt
    
#     cnt += 1
#     #make sure the program will not be killed because of limited usage time
#     if cnt%100 == 0:
#         os.system("keep-job 48")
    
#     global tokenizer
#     return tokenizer(items["text"],max_length=100000)


from pathlib import Path

target_path = "./downloaded_dataset"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_path)
# dataset = load_dataset("the_pile", split="train")

# print("dataset rows :"+str(dataset.num_rows))
# tokenized_dataset = dataset.map(tokenize_map_function, batched = True)

# print("Encoding Completetd")
# print("Vocab Size:"+ str(tokenizer.vocab_size))
# print("Start writing")
# tokenized_dataset.save_to_disk("./tokenized_dataset/tokenized_pile")
tokenized_dataset = load_from_disk("./tokenized_dataset/tokenized_pile")
print(tokenized_dataset)
# Write tokenized Dataset to the binary file
import numpy as np
encodedFile = "./pile_gpt2.bin"
encodedSeqF = open(encodedFile,'wb')


cnt =0 
def write2BinaryFile(items):
    global encodedSeqF
    global cnt
    cnt += 1
    if cnt%100 == 0:
       os.system("keep-job 48")
    for tokens in items['input_ids']:
        tokens_len =int(len(tokens))
        # write its length
        encodedSeqF.write(tokens_len.to_bytes(4,byteorder='little',signed=True))
        
        # write the list of tokens
        encodedSeqF.write(np.array(tokens,dtype=np.uint32).tobytes())

# use map function of dataset to write each tokens into binary file
tokenized_dataset.map(write2BinaryFile, batched = True)
encodedSeqF.close()
