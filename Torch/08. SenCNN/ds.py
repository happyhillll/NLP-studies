# ds.py
from torch.utils.data import Dataset
import torch
from constant import START_TOKEN, END_TOKEN

class SenCLSDataset(Dataset):
    def __init__(self, doc, vocab):
        self.preprocess(doc, vocab)

    def preprocess(self, datas, vocab):
        self.x, self.y = [], []
        for data in datas:
            sent, label = data

            words = sent.split()

            self.x.append([vocab[w] for w in words])
            self.y.append(label)

        self.x = torch.LongTensor(self.x)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]