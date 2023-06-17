# ds.py
from torch.utils.data import Dataset
import torch
import random


class NSMCDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.preprocess(data, tokenizer)

    def preprocess(self, datas, tokenizer):
        """
        :param datas: list of (id, text, label)
        :param tokenizer:
        :return:
        """
        self.x, self.y = [], []

        texts = [d[1] for d in datas]
        labels = [int(d[2]) for d in datas]
        """
        texts, labels = [], []
        for d in datas:
            texts.append(d[1])
            labels.append(int(d[2]))
        """

        tokens = tokenizer(texts, truncation=True, padding="max_length")


        tokens.input_ids = torch.LongTensor(tokens.input_ids)
        tokens.attention_mask = torch.LongTensor(tokens.attention_mask)
        tokens.token_type_ids = torch.LongTensor(tokens.token_type_ids)


        for i in range(len(tokens.input_ids)):
            token = {
                'input_ids': tokens.input_ids[i],
                'attention_mask': tokens.attention_mask[i],
                'token_type_ids': tokens.token_type_ids[i],
            }
            self.x.append(token)

        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]