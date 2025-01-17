import datasets
from datasets import load_from_disk
tokenized_dataset = load_from_disk("./tokenized_dataset/tokenized_pile")
# it = iter(tokenized_dataset)s
for i in range(0,79491):
    tmp_tokens = next(it)
    if(len(tmp_tokens['input_ids'])<=5):
        print(len(tmp_tokens['input_ids']))
#     print(len(tmp_tokens['input_ids']))