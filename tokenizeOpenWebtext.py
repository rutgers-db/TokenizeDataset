import datasets
from datasets import load_dataset
dataset = load_dataset("openwebtext")

print(dataset)

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
print("Gpt2 Tokenizer Vocab Size:"+ str(tokenizer.vocab_size))
print(tokenizer.is_fast)

def tokenize_map_function(items):
    # iterate each item in a batch
    # print(type(items["text"]))
    global tokenizer
    return tokenizer(items["text"],max_length=10000)

tokenized_dataset = dataset.map(tokenize_map_function, batched = True)

print(tokenized_dataset.column_names)
print("Write it into disk")

# save this dataset for future usage
tokenized_dataset.save_to_disk("/tokenized_dataset")

# encodedFile = "openwebtext_gpt2.bin"
# encodedSeqF = open(encodedFile,'wb')


# cnt =0 
# for tokens in tokenized_dataset['train']['input_ids']:
#     tokens_len =int(len(tokens))
#     encodedSeqF.write(tokens_len.to_bytes(4,byteorder='little',signed=True))
#     cnt += 1
#     for i in range(0,tokens_len):
#         encodedSeqF.write((tokens[i]).to_bytes(4,byteorder='little',signed=True))

#     if cnt%10000 == 0:
#         print("Current Written text :"+str(cnt))