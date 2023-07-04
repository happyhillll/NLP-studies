from transformers import AutoModel
from torch.utils.data import DataLoader
import csv
from sklearn.model_selection import train_test_split

# http://air.changwon.ac.kr/?page_id=10
with open('/Users/minjikim/GitHub/NLP-studies/Torch/21. NER using BERT, pytorch lightning/data/label.txt', 'r', encoding='utf-8') as f:
    labels = []
    for line in f:
        labels.append(line.strip())
    print(labels)

filename="/Users/minjikim/GitHub/NLP-studies/Torch/21. NER using BERT, pytorch lightning/data/train.tsv"

with open(filename, 'r', encoding='utf-8') as tsvfile:
    data = []
    reader = csv.reader(tsvfile, delimiter='\t')
    for line in reader:
        data.append(line)
    print(data[0])
    
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    

class ds:
    def __init__(self):
        self.x_train = []
        for i in range(len(train_data)):
            self.x_train.append(train_data[i][0])
        self.y_train = []
        for i in range(len(train_data)):
            self.y_train.append(train_data[i][1])
        
        self.x_val = []
        for i in range(len(val_data)):
            self.x_val.append(val_data[i][0])
        self.y_val = []
        for i in range(len(val_data)):
            self.y_val.append(val_data[i][1])
        self.label = labels
    
    def get_train(self):
        return self.x_train, self.y_train
    
    def get_val(self):
        return self.x_val, self.y_val
    
    def get_label(self):
        return self.label

if __name__ == '__main__':
    print()