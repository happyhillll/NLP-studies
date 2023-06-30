from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch
import lightning as L
from torch.optim.optimizer import Optimizer
from transformers import AutoModel
from bertNERdata import *
from torch.utils.data import DataLoader

class NERClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        device ="mps"
        plm="klue/roberta-base"
        self.bert = AutoModel.from_pretrained(plm).to(device) #pretrained 모델(plm에 쥐어준 모델) 불러오기
        #drop-out : 0부터 1사이의 확률로 뉴런을 제거하는 기법 _ 특성 feature 만을 과도하게 학습하는 오버피팅을 방지하기 위함
        self.dropout = nn.Dropout(0.1).to(device) # 뉴런별로 0.1의 확률로 drop 될지 여부가 결정됨
        self.classifier = nn.Linear(768, 30).to(device) #BERT의 임베딩 차원은 768임
        self.loss_fct = nn.CrossEntropyLoss().to(device) #MLM을 위한 loss function
        
    def forward(self, batch): # forward : logit 출력을 위한
        #각각 GPU로 조금씩 넘겨줌
        outputs = self.bert(**batch).to(self.device) # **batch : dict 형태로 unpacking
        pooled_output = self.dropout(outputs.pooler_output).to(self.device) #dropout 하기
        logits = self.classifier(pooled_output).to(self.device)    # [32,768] -> [32,2]

        return logits
    
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input,labels=train_batch
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