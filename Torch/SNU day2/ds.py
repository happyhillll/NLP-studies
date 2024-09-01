import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, x, y=None, normalize=True):
        super(CustomDataset, self).__init__()
        self.x = x
        self.y = y

        # 데이터 표준화
        if normalize:
            scaler = StandardScaler()
            
            # x를 numpy 배열로 변환
            if isinstance(self.x, pd.DataFrame):
                self.x = scaler.fit_transform(self.x.values)  # DataFrame의 값을 numpy 배열로 변환하여 표준화
            elif isinstance(self.x, pd.Series):
                self.x = scaler.fit_transform(self.x.values.reshape(-1, 1))  # Series를 numpy 배열로 변환 후 reshape

        # 텐서 변환
        self.x = torch.tensor(self.x).float()

        if y is not None:
            if isinstance(self.y, pd.DataFrame):
                self.y = torch.tensor(self.y.values).float()
            elif isinstance(self.y, pd.Series):
                self.y = torch.tensor(self.y.values.reshape(-1, 1)).float()
            else:
                self.y = torch.tensor(self.y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.y is not None:
            y = self.y[idx]
            return x, y
        else:
            return x

