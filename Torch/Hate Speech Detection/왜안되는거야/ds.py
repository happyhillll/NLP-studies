from torch.utils.data import Dataset #torch.utils.data.Dataset을 가져옴
import torch
import random 
from data import ds

class HateDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.preprocess(data, tokenizer)
    
    def preprocess(self, data, tokenizer):
        texts=ds().get_train()[0]
        labels=ds().get_dev()[1]
        
        tokens=tokenizer(texts, truncation=True, padding='max_length')
        
        tokens.input_ids=torch.LongTensor(tokens.input_ids)
        tokens.attention_mask=torch.LongTensor(tokens.attention_mask)
        tokens.token_type_ids=torch.LongTensor(tokens.token_type_ids)
        
        self.x, self.y=[], []
        
        for i in range(len(tokens.input_ids)):
            token={
                'input_ids': tokens.input_ids[i],
                'attention_mask': tokens.attention_mask[i],
                'token_type_ids': tokens.token_type_ids[i],
            }
            self.x.append(token)
        
        self.y=torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]