import torch.nn as nn

class BigramLM(nn.Module):
    def __init__(self, vocab_len, dim):
        super().__init__()
        self.tabel = nn.Embedding(num_embeddings=vocab_len, embedding_dim=dim)
        self.linear=nn.Linear(2*dim, vocab_len) #2*dim : the concatenation of two words
        #vocab_len : the probability distribution over the entire vocabulary of words
        
    def forward(self, x):
        # x : [batch_size, 2]
        emb = self.tabel(x) # [batch_size, 2, dim]
        bsz = emb.shape[0] #batch_size
        emb = emb.view(bsz, -1) # [batch_size, 2*dim]
        pred = self.linear(emb) # [batch_size, vocab_len]
        
        return pred
    
    
class CNNforsentclass(nn.Module):
    def __init__(self, vocab_len,dim))
        super().__init__()
        self.embedding=nn.Embedding(vocab_len,dim)
        