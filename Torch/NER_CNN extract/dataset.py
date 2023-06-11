#dataset.py
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
mps_device = torch.device("mps")
import numpy as np
from datasets import load_dataset
import numpy as np
from vocab import get_character

dataset = load_dataset("klue", "ner")

#max_len 은 145임
# lengths = np.array([len(sentence) for sentence in dataset["train"]["tokens"]])
# max_length=np.max(lengths)
# print(max_length)

class CNNextractDataset(Dataset):
    def __init__(self, datas, char):
        self.preprocess(datas, char)
        
    def padding(self,xs,max_len=145):
        asd
    
    def preprocess(self, datas, char):
        self.x,self.y=[],[]
        for data in datas:
            sent, labels = data
            
            
            self.x.append([char[w] for w in character])
            self.y.append(torch.tensor(label))
            
        max_len=145
        padded = []
        for sent in self.x:
            if len(sent) == max_len:
                padded.append(sent)
            elif len(sent) > max_len:
                padded.append(sent[:max_len])
            else:  # len(sent) < max_len
                new_sent = sent + [0]*(max_len-len(sent))
                padded.append(new_sent)

        self.x = padded
        
        #y도 똑같이
        self.y = padded
        
        return self.x, self.y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]