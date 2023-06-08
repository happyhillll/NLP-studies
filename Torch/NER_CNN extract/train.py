#train.py
import torch.nn as nn
import torch
from dataset import CNNextractDataset
from torch.utils.data import DataLoader
from ds import *
from torch.utils.data import random_split

print(torch.__version__)
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
mps_device = torch.device("mps")
dataset = ds()
x_train, y_train = dataset.get_train()
data = list(zip(x_train, y_train))