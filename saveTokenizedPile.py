import datasets
from datasets import load_from_disk
tokenized_dataset = load_from_disk("./tokenized_dataset/tokenized_pile")
it = iter(tokenized_dataset)
import numpy as np
encodedFile = "./pile_gpt2_test.bin"
encodedSeqF = open(encodedFile,'wb')

buffer = []
for i in range(0, tokenized_dataset.num_rows):
    tokens = next(it)['input_ids']
    buffer.append(len(tokens))
    buffer.extend(tokens)
    
    if len(buffer) > 100000000:
        encodedSeqF.write(np.array(buffer,dtype = np.uint32).tobytes())
        # write the list of tokens
        buffer = []
        print(str(i)+"/"+str(tokenized_dataset.num_rows))

encodedSeqF.write(np.array(buffer,dtype = np.uint32).tobytes())
encodedSeqF.close()   
