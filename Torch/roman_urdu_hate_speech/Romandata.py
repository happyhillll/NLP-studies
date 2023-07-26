#data.py
from torch.utils.data import dataset
import csv

#집 : /Users/minjikim/GitHub/NLP-studies/Torch/roman_urdu_hate_speech/data/task_1_train.tsv

#train
filename='/Users/minjikim/GitHub/NLP-studies/Torch/roman_urdu_hate_speech/data/task_1_train.tsv'
f=open(filename,'r',encoding='utf-8')
rdr=csv.reader(f,delimiter='\t')
train=list(rdr)

f.close()

#val
filename='/Users/minjikim/GitHub/NLP-studies/Torch/roman_urdu_hate_speech/data/task_1_validation.tsv'
f=open(filename,'r',encoding='utf-8')
rdr=csv.reader(f,delimiter='\t')
val=list(rdr)

f.close

class ds:
    def __init__(self):
        self.x_train=[]
        for i in range(len(train)):
            self.x_train.append(train[i][0])
        self.y_train=[]
        for i in range(len(train)):
            self.y_train.append(train[i][1])
        
        self.x_val=[]
        for i in range(len(val)):
                self.x_val.append(val[i][0])
        self.y_val = []
        for i in range(len(val)):
            self.y_val.append(val[i][1])
    
    def get_train(self):
        return self.x_train, self.y_train
    
    def get_val(self):
        return self.x_val, self.y_val
    
if __name__ == '__main__':
    print()
        