import torch.nn as nn
import torch

INPUT_DIM = 2250 # vocab size
EMBEDDING_DIM = 100 # embedding size
HIDDEN_DIM = 512 
OUTPUT_DIM = 1 # binary classification

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

'''
encoder : 문장의 의미 출력
decoder : 진짜 output 출력

LSTM 어텐션 써보기

bmm = softmax 랑 encoder의 각각 아웃풋이랑 곱해주기

Q : 이 질문에 대해서 어느정도 유사성을 가지고 있나?
query랑 key의 유사도를 구해서 > 얼만큼 가중치를 줄건지 정함
쿼리가 키한테 얼마나 유사한지 물어봄.  니가 얼마나 여기에 관련성이 있냐?
key랑 value랑 보통 똑같음 왜냐면 얼마나 연관성이 있다는 걸 의미하는게 value이니까
key는 linear1과 ㅣinear2를 각각 더해서 나온거임

'''