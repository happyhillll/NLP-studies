# data.py
# from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset

dataset=load_dataset('klue', 'mrc')
print(dataset['train'][0])

class ds:
    def __init__(self):
        self.x_train_q = []
        self.x_train_context = []
        self.y_train = []
        for i in range(len(dataset['train'])):
            self.x_train_q.append(dataset['train'][i]['question'])
            self.x_train_context.append(dataset['train'][i]['context'])
            self.y_train.append(dataset['train'][i]['answers']['text'][0])
    
        self.x_val_q = []
        self.x_val_context = []
        self.y_val = []
        for i in range(len(dataset['validation'])):
            self.y_train_q.append(dataset['validation'][i]['question'])
            self.y_train_context.append(dataset['validation'][i]['context'])
            self.y_train.append(dataset['validation'][i]['answers']['text'][0])
    
    def get_train(self):
        return self.x_train_q, self.x_train_context, self.y_train
    
    def get_val(self):
        return self.x_val_q, self.x_val_context, self.y_val
    
    