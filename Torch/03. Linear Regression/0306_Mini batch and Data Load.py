import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

x_train = torch.FloatTensor([[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

dataset=TensorDataset(x_train,y_train)
dataloader=DataLoader(dataset, batch_size=2, shuffle=True)

