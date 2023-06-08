#model.py
import torch.nn as nn
import torch.nn.functional as F 
import torch
mps_device = torch.device("mps")


class LSTM(nn.Module):
    def __init__(self,vocab_len, dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=dim, padding_idx=vocab_len -1) #padding_idx는 패딩 인덱스를 저장하는 정수, 입력 텍스트를 특정 길이로 맞추기 위해 사용
        self.lstm = nn.LSTM(dim, hidden_size= hidden_dim, num_layers= num_layers, bidirectional=True, batch_first=True, dropout=dropout)  
        self.tanh1 = nn.Tanh() #non-linear function
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2)) # 2 for bidirection
        self.fc1 = nn.Linear(hidden_dim *2, 64) # 64 for attention 일반적으로 사용하는 숫자
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        emb = self.embedding(x)
        output, _  = self.lstm(emb) # output: [batch_size, seq_len, hidden_dim * 2]
        M = self.tanh1(output) # M: [batch_size, seq_len, hidden_dim * 2] input sequence에 대한 정보를 담고 있음
        # hidden state에 얼만큼의 가중치를 두어야할지 결정함
        # alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        alpha = F.softmax((M * self.w).sum(dim=2), dim=1).unsqueeze(-1)  
        #  M * self.w: input sequence * attention weights = the weighted sum of the input sequence (importance of each word)
        #  sum(dim=2): sum of each word in the input sequence
        #  softmax: normalize the weighted sum of the input sequence
        #  unsqueeze(-1): add one dimension at the end of the tensor
        out = output * alpha  
        out = torch.sum(out, 1) # out: [batch_size, hidden_dim * 2] > [batch_size, 1]
        out = F.relu(out)
        out = self.fc1(out) 
        out = self.fc(out)
        
        return out

'''
vocab_len: vocab 개수
dim: embedding 차원
hidden_dim: lstm hidden 차원
num_layers: lstm layer 개수
dropout: dropout 비율
'''
'''
1. 임베딩
2. lstm
3. tanh
4. attention
5. fc
'''