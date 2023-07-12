from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch
import lightning as L
from torch.optim.optimizer import Optimizer
from transformers import AutoModel
from ds import *
from torch.utils.data import DataLoader

class NER_classifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        device ="mps"
        plm="klue/roberta-base"
        self.bert = AutoModel.from_pretrained(plm).to(device) #pretrained 모델(plm에 쥐어준 모델) 불러오기
        self.dropout = nn.Dropout(0.1).to(device)
        self.classifier = nn.Linear(768, 30).to(device)
        self.loss_fct = nn.CrossEntropyLoss().to(device)
    
    def forward(self, batch):
        outputs = self.bert(**batch).to(self.device)
        pooled_output = self.dropout(outputs.pooler_output).to(self.device)
        logits=self.classifier(pooled_output).to(self.device)
        return logits
    
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        input, labels=train_batch
        input['input_ids'] = input['input_ids'].squeeze() 
        input['attention_mask'] = input['attention_mask'].squeeze()
        input['token_type_ids'] = input['token_type_ids'].squeeze()

        logits = self.bert(input)
        loss = self.loss_fct(logits, labels)
        self.log('train_loss',loss,on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        input, labels=val_batch
        input['input_ids'] = input['input_ids'].squeeze()
        input['attention_mask'] = input['attention_mask'].squeeze()
        input['token_type_ids'] = input['token_type_ids'].squeeze()

        logits = self.bert(input)
        loss = self.loss_fct(logits, labels)
        self.log('val_loss', loss, on_epoch=True)
        return loss
    
    def backward(self,trainer,loss,optimizer,optimizer_idx):
        loss.backward()

    def optimizer_step(self,epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
        
    