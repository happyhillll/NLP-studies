# import torch
# from torchtext import data
# from torchtext.vocab import Vectors # 단어 벡터를 가져오기 위한 모듈
# #https://github.com/Kyubyong/wordvectors

# MAX_VOCAB_SIZE = 25000

# ko_vectors = Vectors(name='ko.vec', cache='/Users/minjikim/GitHub/NLP-studies/Torch/RNN/ko') # 한국어 Word2Vec 모델을 가져온다.
    
# print(ko_vectors)
import torch
import urllib.request
import pandas as pd
from torchtext import data
from torchtext import datasets

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

train_data['document']=train_data['document'].astype(str)
train_data['document_split']=train_data['document'].apply(lambda x:x.split())
train_data_drop=train_data[['document_split','label']]

'''
학습 완료..
그담에 어떻게 해야하는지 모르겠음
'''

def get_w2v():
    #임베딩 모델 만들기
    from gensim.models import Word2Vec

    embedding_model = Word2Vec(train_data_drop['document_split'],vector_size=100, window = 2, min_count=50, workers=4, sg=1)
    print(embedding_model.wv.vectors.shape)
    vocab=embedding_model.wv.vectors
    return dict(zip(embedding_model.wv.key_to_index,embedding_model.wv.vectors))

# #임베딩 모델에서 "영화"와 유사한 단어와 벡터값을 model_result에 저장
# model_result=embedding_model.wv.most_similar("영화")
# print(model_result)

# #임베딩 모델 저장 및 로드
# from gensim.models import KeyedVectors

# embedding_model.wv.save_word2vec_format('/Users/minjikim/GitHub/NLP-studies/Torch/RNN/naver_embedding_model')
# loaded_model=KeyedVectors.load_word2vec_format("/Users/minjikim/GitHub/NLP-studies/Torch/RNN/naver_embedding_model")

