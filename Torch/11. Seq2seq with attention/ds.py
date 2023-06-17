# ds.py
from torch.utils.data import Dataset
import torch



class nsmc_SentDataset(Dataset):
    def __init__(self, doc, vocab):
        self.preprocess(doc, vocab)

    def preprocess(self, datas, vocab):
        self.x, self.y = [], []
        for data in datas:
            document, label = data

            words = document.split()
            total_length = 20
            
            self.x = [vocab[w] for w in words] + [0]*(total_length - len(words))
            
        self.x=torch.LongTensor(self.x)
        self.y=torch.LongTensor(label)
            # x = longTensor(x)
            # self.x.append([vocab[w] for w in words]) 
        #     self.y.append(label)
            
        # self.x = torch.zeros(      
        # self.x = torch.LongTensor(self.x)
        # self.y = torch.LongTensor(self.y)
        # #토큰화하고 패딩도 하고 preprocess에 하거나
        
    #혹은 여기 def를 만들어서 하거나

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]