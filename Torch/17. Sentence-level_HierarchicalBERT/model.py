# model.py

import torch.nn as nn
import torch
from transformers import AutoModel

class LongDocClassifier(nn.Module):
    def __init__(self, plm, device):
        super().__init__()
        self.device = device
        self.bert = AutoModel.from_pretrained(plm).to(device)

        self.dropout = nn.Dropout(0.1).to(device)
        self.linear_1 = nn.Linear(768, 100).to(device)
        self.lstm = nn.LSTM(100, 100, num_layers=2, batch_first=True).to(device)

        self.linear_2 = nn.Linear(100, 30).to(device)
        self.gelu = nn.GELU()
        self.linear_3 = nn.Linear(30, 17).to(device)


    def forward(self, batch):

        batch["input_ids"] = batch["input_ids"].to(self.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)

        outputs = self.bert(**batch)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.linear_1(pooled_output)    # [4,768] -> [4,100]
        logits = logits.unsqueeze(0)
        outputs, _ = self.lstm(logits)  # [1, 4, 100]
        emb = outputs[:,-1,:]  # [1, 100]
        return self.linear_3(self.gelu(self.linear_2(emb)))  # [17]