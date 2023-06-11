#model.py
import torch.nn as nn
import torch.nn.functional as F 
import torch
mps_device = torch.device("mps")

class CNNextract(nn.Module):
    def __init__(self):
        super().__init__()
        n_filter = 100 # 필터 개수는 100개 : 서로 다른 특성을 잡아주는 필터들
        self.static=nn.Embedding(num_embeddings=1968, embedding_dim=100, _freeze=True) #static channel, 50개의 단어를 300차원씩
        #num_embeddings : size of the dictionary of embeddings, embedding_dim : the size of each embedding vector
        self.conv=nn.Conv1d(in_channels=300, out_channels=n_filter, kernel_size=5) #in_channels : input의 feature dimension, out_channel:내가 output으로 내고싶은 dimension
        self.linear=nn.Linear(n_filter, 13)

    def forward(self,x): #x:[bsz, seq_len]
        emb = self.static(x) #[bsz, seq_len, 300]
        emb = emb.permute(0,2,1) #[bsz, 300, seq_len]
        output=self.conv(emb) #[bsz, n_filter(100), num_sliding]
        pooled = torch.max(output,2)[0] #[bsz, n_filter] 2번째 있는걸 max로 해주고, 나온 값중에 0번째 값을 저장해라. [1]: max의 인덱스를 뽑아줌
        
        y_hat = self.linear(pooled) #[bsz,n_cls] 
        
        return y_hat