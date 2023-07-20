import lightning as L
from transformers import AutoModel
import torch.nn as nn
import torch

class hateClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        plm="klue/roberta-base"
        self.bert=AutoModel.from_pretrained(plm)
        self.dropout=nn.Dropout(0.1) #뉴런별로 0.1의 확률로 drop 될지 여부가 결정됨
        self.classifier=nn.Linear(768,2) #BERT의 임베딩 차원은 768임
        self.loss_fct=nn.CrossEntropyLoss() #MLM을 위한 loss function
    
    def forward(self,batch):
        outputs=self.bert(**batch) # **batch : dict 형태로 unpacking
        pooled_output=self.dropout(outputs.pooler_output) #dropout 하기
        logits=self.classifier(pooled_output)   # [32,768] -> [32,2]
        
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
        input, labels=val_batch #이미 2개가 왔는데 왜 된다는건지?
        input['input_ids'] = input['input_ids'].squeeze()
        input['attention_mask'] = input['attention_mask'].squeeze()
        input['token_type_ids'] = input['token_type_ids'].squeeze()

        logits = self.bert(input)
        loss = self.loss_fct(logits, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.accuracy(logits, labels), prog_bar=True)
        return {"loss":loss,"pred":logits,"label":labels}