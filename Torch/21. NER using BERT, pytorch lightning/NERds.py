# ds.py
from torch.utils.data import Dataset
import torch
import random
from bertNERdata import *
'''
레이블은 워드 단위로 되어져 있음. token 단위로 해석해야함.
1. 필요한거
- char to word [[0 0 1 1 1 1 2 2 2...]]
- word
- labels dict 형태로
- 
'''


class NERDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.preprocess(data, tokenizer) #Dataset을 실행하면 자동으로 preprocess 실행
    
    def preprocess(self, datas, tokenizer):
        texts=ds().get_train()[0][:1000]
        label=ds().get_train()[1][:1000]
        labels=list(set(ds().get_train()[1]))
        
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id
        
        #레이블 숫자 매겨주기
        labels = label.split()
        labels_idx=[]
        for i in labels:
            labels_idx.append(self.labels_lst.index(label) if label in self.labels_lst else self.labels_lst.index("UNK"))
        
         # Tokenize word by word (for NER)
        tokens = []
        label_ids = []
        for word, slot_label in zip(texts, label):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([int(slot_label)] + [unk_token] * (len(word_tokens) - 1))

        
        # vocab={
        #     word : idx for idx, word in enumerate(labels)
        # }
        # labelss=[]
        # for i in label:
        #     if i in vocab.keys():
        #         labelss.append(vocab[i])
        
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

if __name__ == '__main__':
    print(NERDataset(ds().get_train(), tokenizer))