import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
print(len(train_data))
test_data = pd.read_table('ratings_test.txt')

''''
label에 무슨 값이 있는지 확인
train_data['label'].unique() : 0,1만 출력

value=train_data['label'].unique
print(value) : 값 모두 출력

모든 메소드에 대해서 이런식으로 적용가능한지?

nn.embedding 으로 써서 임베딩시키기
'''
