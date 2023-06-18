# ds.py
from torch.utils.data import Dataset
import torch
import random

class NSMCDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.preprocess(data, tokenizer) #Dataset을 실행하면 자동으로 preprocess 실행

    def preprocess(self, datas, tokenizer):
        """
        :param datas: list of (id, text, label)
        :param tokenizer:
        :return:
        """
        texts = [d[1] for d in datas] #데이터 안에 1번째 열을 텍스트로 지정
        labels = [int(d[2]) for d in datas] #데이터 안에 2번째 열을 레이블(감성)으로 지정
        """
        texts, labels = [], []
        for d in datas:
            texts.append(d[1])
            labels.append(int(d[2]))
        """

        tokens = tokenizer(texts, truncation=True, padding="max_length") #텍스트 토크나이징


        tokens.input_ids = torch.LongTensor(tokens.input_ids) #텍스트 ids 
        tokens.attention_mask = torch.LongTensor(tokens.attention_mask) #패딩 토큰 구분
        tokens.token_type_ids = torch.LongTensor(tokens.token_type_ids) #문장 구분

        self.x,self.y=[],[]

        for i in range(len(tokens.input_ids)):
            token = {
                'input_ids': tokens.input_ids[i],
                'attention_mask': tokens.attention_mask[i],
                'token_type_ids': tokens.token_type_ids[i],
            }
            self.x.append(token) #x에는 token이라는 dict를 넣어줌

        self.y = torch.LongTensor(labels) #y에는 labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]