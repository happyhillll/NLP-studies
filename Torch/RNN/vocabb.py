import torch
from torchtext import data
from torchtext.vocab import Vectors # 단어 벡터를 가져오기 위한 모듈
#https://github.com/Kyubyong/wordvectors

MAX_VOCAB_SIZE = 25000

ko_vectors = Vectors(name='ko.vec', cache='/Users/minjikim/GitHub/NLP-studies/Torch/RNN/ko') # 한국어 Word2Vec 모델을 가져온다.
    
print(ko_vectors)

