# data.py
# from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset

dataset=load_dataset('klue', 'mrc')
print(dataset['train'][0])

class ds:
    def __init__(self):
        self.train_question = []
        self.train_context = []
        self.train_answer = []
        for i in range(len(dataset['train'])):
            self.train_question.append(dataset['train'][i]['question'])
            self.train_context.append(dataset['train'][i]['context'])
            self.train_answer.append(dataset['train'][i]['answers'])
    
        self.val_question = []
        self.val_context = []
        self.val_answer = []
        for i in range(len(dataset['validation'])):
            self.val_question.append(dataset['validation'][i]['question'])
            self.val_context.append(dataset['validation'][i]['context'])
            self.val_answer.append(dataset['validation'][i]['answers'])
    
    def get_train(self):
        return self.train_question, self.train_context, self.train_answer
    
    def get_val(self):
        return self.val_question, self.val_context, self.val_answer
# if __name__ == '__main__':
#     ds=ds()

print(2462+2511+2567+2598)