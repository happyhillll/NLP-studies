import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

print(len(train_data)) # 훈련용 리뷰 개수 출력

from konlpy.tag import Okt, Komoran, Hannanum, Kkma , Mecab

okt = Okt()
komoran = Komoran()
hannanum = Hannanum()
kkma = Kkma()
mecab = Mecab()


tagger = Mecab()
tagger = tagger.morphs

import torch
from torchtext import data
from torchtext import datasets


SEED = 1234

torch.manual_seed(SEED)

DOCUMENT = data.Field(tokenize = tagger ,include_lengths = False)
LABEL = data.LabelField(dtype = torch.float)

train_data={'document':('document',DOCUMENT),'label':('label', LABEL) } #dictionary 생성
print(dic)

''''
label에 무슨 값이 있는지 확인
train_data['label'].unique() : 0,1만 출력

value=train_data['label'].unique
print(value) : 값 모두 출력

모든 메소드에 대해서 이런식으로 적용가능한지?
'''

