# 모델 설계
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear=nn.Linear(in_dim,out_dim)
    
    def forward(self,x):
        return self.linear(x)