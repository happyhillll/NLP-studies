# ds.py
from torch.utils.data import Dataset
import torch
import random
from data import ds

class hateDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.preprocess(data, tokenizer) #Dataset을 실행하면 자동으로 preprocess 실행
    
    def preprocess(self, datas, tokenizer):
        
        texts=ds().get_train()[0][:5000]
        label=ds().get_train()[1][:5000]
        labels=list(set(ds().get_train()[1]))
        vocab={
            word : idx for idx, word in enumerate(labels)
        }
        labelss=[]
        for i in label:
            if i in vocab.keys():
                labelss.append(vocab[i])
                
        
        # vocab={
        #     word : idx for idx, word in enumerate(labels)
        # }
        # labelss=[]
        # labelss.append(vocab[x] for x in label)
        
        tokens=tokenizer(texts, truncation=True, padding='max_length')
        
        tokens.input_ids = torch.LongTensor(tokens.input_ids)
        tokens.attention_mask = torch.LongTensor(tokens.attention_mask)
        tokens.token_type_ids = torch.LongTensor(tokens.token_type_ids)
        
        self.x, self.y = [], []
        
        for i in range(len(tokens.input_ids)):
            token={
                'input_ids': tokens.input_ids[i],
                'attention_mask': tokens.attention_mask[i],
                'token_type_ids': tokens.token_type_ids[i],
            }
            self.x.append(token)
        
        self.y = torch.LongTensor(labelss)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]