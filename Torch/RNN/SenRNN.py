import torch.nn as nn
import torch

INPUT_DIM = len(DOCUMENT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 512
OUTPUT_DIM = 1

print(INPUT_DIM)
print(OUTPUT_DIM)

class SenRNN(nn.Module):
  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.rnn = nn.RNN(embedding_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self,text):
    #text = [sent len, batch_size]
    embedded = self.embedding(text)
    #embedded = [sent len. batch size, emb dim]
    output, hidden = self.rnn(embedded)

    #output = [sent len, batch size, hid dim]
    #hidden = [1, batch size, hid dim]
    assert torch.equal(output[-1,:,:],hidden.squeeze(0))
    return self.fc(hidden.squeeze(0))
