from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
from data import *
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data, target="target", normalize=True):
        super(CustomDataset, self).__init__()
        self.x = data.drop(target, axis=1)

        # 데이터 표준화
        if normalize:
            scaler = StandardScaler()
            self.x = pd.DataFrame(scaler.fit_transform(self.x))

        self.y = data[target]

        # 텐서 변환
        self.x = torch.tensor(self.x.values).float()
        self.y = torch.tensor(self.y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
    