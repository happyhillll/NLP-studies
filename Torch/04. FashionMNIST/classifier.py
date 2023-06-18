import torch.nn as nn

class Fashion_MNIST_Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear=nn.Linear(in_dim,out_dim)
    
    def forward(self,x):
        return self.linear(x)