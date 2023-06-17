from torch import nn
import torch
import torch.nn.functional as F

device = None

#attention based bidirectional LSTM model
class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_avocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2)) #제로로 초기화
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc = nn.Linear(64, config.num_classes)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256] 
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1] 스칼라값에 softmax를 씌우면 alpha가 나옴. 
        out = H * alpha  # [128, 32, 256] n=32
        out = torch.sum(out, 1)  # [128, 256] #곲한걸 elment wise로 더함
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
    
    '''
    Energy : softmax 먹이기 이전의 변수
    
    Q* tanh = linear로 퉁칠 수 있음
    0.2 * 3 = 0.6
    어차피 non linear를 씌울거니까 .. 그치만 일반적으로 Q K V ㄹ르 나눔
    
    내가 얼마만큼 ㄱ관련이 있냐, Q랑 곱해지는 애가 K라고 보면 됨.
    Q는 항상 ㄴ하나고 key는 ㅍ미ㅕㄷ rotn
    
    value : 최종 아웃풋 낼때 
    
    '''
    
    
    ''''
    z=encode(src)
    output=decode(z)
    
    mutable
    immutable
    
    '''