# ds.py
from torch.utils.data import Dataset
import torch
from constant import START_TOKEN, END_TOKEN

class BiGramDataset(Dataset):
    def __init__(self, doc, vocab):
        self.preprocess(doc, vocab)


    def preprocess(self, doc, vocab):
        words = doc.split()
        words = [START_TOKEN, START_TOKEN] + words + [END_TOKEN]
        self.x, self.y = [], []

        for end_idx in range(2, len(words)):
            xs = words[end_idx-2: end_idx] # 앞에 두개
            y = words[end_idx] # 뒤에 하나

            xs = [vocab[x] for x in xs] # 
            self.x.append(xs)
            self.y.append(vocab[y])

        self.x = torch.LongTensor(self.x)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]