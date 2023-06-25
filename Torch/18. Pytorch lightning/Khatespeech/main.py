from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch
import lightning as L
from torch.optim.optimizer import Optimizer
from transformers import AutoModel
from hatedata import *
from torch.utils.data import DataLoader
dss=ds()

train_data = dss.get_train()
train_loader = DataLoader(train_data, batch_size=2)

plm="klue/roberta-base"

class hateClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(plm) #pretrained 모델(plm에 쥐어준 모델) 불러오기
        #drop-out : 0부터 1사이의 확률로 뉴런을 제거하는 기법 _ 특성 feature 만을 과도하게 학습하는 오버피팅을 방지하기 위함
        self.dropout = nn.Dropout(0.1) # 뉴런별로 0.1의 확률로 drop 될지 여부가 결정됨
        self.classifier = nn.Linear(768, 3) #BERT의 임베딩 차원은 768임
        self.loss_fct = nn.CrossEntropyLoss() #MLM을 위한 loss function
        
    def forward(self, batch): # forward : logit 출력을 위한
        #각각 GPU로 조금씩 넘겨줌
        outputs = self.bert(**batch) # **batch : dict 형태로 unpacking
        pooled_output = self.dropout(outputs.pooler_output) #dropout 하기
        logits = self.classifier(pooled_output)    # [32,768] -> [32,2]

        return logits
    
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters, lr=1e-3)
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

    def backward(self,trainer,loss,optimizer,optimizer_idx):
        loss.backward()

    def optimizer_step(self,epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()

model=hateClassifier()
trainer=L.Trainer()
trainer.fit(model, train_loader)