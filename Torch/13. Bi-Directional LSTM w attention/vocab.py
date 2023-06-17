#vocab.py
from constant import START_TOKEN, END_TOKEN
from ds import *

def get_vocab():
    
    dataset = ds()
    x_train, y_train = dataset.get_train()
    
    # for i, text in enumerate(x_train):
    #     if not isinstance(text, str):
    #         print(f'Item {i} in x_train is not a string: {text}')
        
    # words = [START_TOKEN, END_TOKEN] + list(set(word for text in x_train for word in text.split()))
    # x_train 리스트에서 모든 문장을 가지고온 후 split()으로 단어를 분리하여 words 리스트에 저장하고 start_token과 end_token을 추가한다.
    # 이때 각 단어가 문자열인지 확인하고,  set을 이용해 중복된 단어는 제거한다.
    words = [START_TOKEN, END_TOKEN] + list(set(word for text in x_train if isinstance(text, str) for word in text.split()))

    vocab = {
        word: idx for idx, word in enumerate(words)
    }
    # id_to_word = {
    #     v: k for k, v in word_to_id.items()
    # }

    return vocab