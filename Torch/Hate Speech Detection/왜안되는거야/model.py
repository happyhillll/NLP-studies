import lightning as L
from transformers import AutoModel
import torch.nn as nn
import torch

class hateClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        device="cpu"
        plm="klue/roberta-base"
        self.bert=AutoModel.from_pretrained(plm).to(device)
        self.dropout=nn.Dropout(0.1).to(device) #뉴런별로 0.1의 확률로 drop 될지 여부가 결정됨
        self.classifier=nn.Linear(768,2).to(device) #BERT의 임베딩 차원은 768임
        self.loss_fct=nn.CrossEntropyLoss().to(device) #MLM을 위한 loss function
    
    def forward(self,batch):
        outputs=self.bert(**batch).to(self.device) # **batch : dict 형태로 unpacking
        pooled_output=self.dropout(outputs.pooler_output).to(self.device) #dropout 하기
        logits=self.classifier(pooled_output).to(self.device)    # [32,768] -> [32,2]
        
        return logits #logits : [32,2]
        
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self,train_batch, batch_idx):
        input,labels=train_batch
        input['input_ids']=input['input_ids'].squeeze()
        input['attention_mask']=input['attention_mask'].squeeze()
        input['token_type_ids']=input['token_type_ids'].squeeze()
        
        logits=self.bert(input)
        loss=self.loss_fct(logits,labels) 
        self.log('train_loss',loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.accuracy(logits, labels), prog_bar=True) 
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        input, labels=val_batch
        input['input_ids'] = input['input_ids'].squeeze()
        input['attention_mask'] = input['attention_mask'].squeeze()
        input['token_type_ids'] = input['token_type_ids'].squeeze()

        logits = self.bert(input)
        loss = self.loss_fct(logits, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.accuracy(logits, labels), prog_bar=True)
        return {"loss":loss,"pred":logits,"label":labels}