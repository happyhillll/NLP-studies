# data.py
# from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset

dataset=load_dataset('klue', 'mrc')
print(dataset['train'][0])

class ds:
    def __init__(self):
        self.get_qa_data()
        self.x_train_q = []
        self.x_train_context = []
    
    def get_qa_data():
        dataset = load_dataset('klue', 'mrc')
        return dataset['train'], dataset['validation']
    
    def get_train(self):
        self.x_train_q.append(dataset['train'][0]['question'])
        self.x_train_context.append(dataset['train'][0]['context'])
        return self.x_train_q, self.x_train_context
    