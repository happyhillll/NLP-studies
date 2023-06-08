# ds.py
# 데이터 불러오고, x_train, y_train, x_test, y_test 리스트로 만들기
from datasets import load_dataset
import numpy as np

dataset = load_dataset("klue", "ner")
#dataset['train'][0]['sentence'], dataset['train'][0]['tokens'], dataset['train'][0]['ner_tags']

ner_tags = list(set([tag for doc in dataset['train'] for tag in doc['ner_tags']]))
ner_tags

class ds:
    def __init__(self):
        self.x_train=list(dataset['train']['tokens'])[:5000]
        self.y_train=list(dataset['train']['ner_tags'])[:5000]
        
        self.x_test=list(dataset['validation']['tokens'])[:1000]
        self.y_test=list(dataset['validation']['ner_tags'])[:1000]
    
    def get_train(self):
        return self.x_train, self.y_train
    
    def get_test(self):
        return self.x_test, self.y_test