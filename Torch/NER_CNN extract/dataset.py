#dataset.py
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
mps_device = torch.device("mps")
import numpy as np

#max_len 은 145임
# lengths = np.array([len(sentence) for sentence in dataset["train"]["tokens"]])
# max_length=np.max(lengths)
# print(max_length)

class CNNextractDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.preprocess(x_train, y_train)
    
    def preprocess(self):
        max_len=145
        self.x_train = [torch.tensor(seq + [0]*(max_len - len(seq))) if len(seq) < max_len else torch.tensor(seq[:max_len]) for seq in self.x_train]
        self.y_train =[torch.tensor(seq + [0]*(max_len - len(seq))) if len(seq) < max_len else torch.tensor(seq[:max_len]) for seq in self.y_train]
    
    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]