from torch.utils.data import Dataset
import torch

#Custom Dataset 만들기
class BiGramDataset(Dataset):
    def __init__(self):
        self.preprocess()
    
    def preprocess(self)
    
    def __len__(self):
        return self.x.shape[0] #? 이거 무슨 의미인지 모르겠
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

