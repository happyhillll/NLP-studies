#model.py
import torch.nn as nn
import torch.nn.functional as F 
import torch
mps_device = torch.device("mps")

class CNNextract(nn.Module):
    def __init__(self):
        super().__init__()
        n_filter = 100 # 필터 개수는 100개 : 서로 다른 특성을 잡아주는 필터들
        self.static=nn.Embedding(num_embeddings=50, embedding_dim=300, _freeze=True) #static channel, 50개의 단어를 300차원씩
        #num_embeddings : size of the dictionary of embeddings, embedding_dim : the size of each embedding vector
        self.nonstatic=nn.Embedding(num_embeddings=50, embedding_dim=300)
        self.conv=nn.Conv1d(in_channels=300, out_channels=n_filter, kernel_size=2) #in_channels : input의 feature dimension, out_channel:내가 output으로 내고싶은 dimension
        self.linear=nn.Linear(n_filter*2, 2)