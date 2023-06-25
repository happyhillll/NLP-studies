import torch.nn as nn
import torch
print(torch.backends.mps.is_available())
from transformers import AutoModel

class hateClassifier(nn.Module):
    def __init__(self, plm, device): #plm : pretrained language model, device : GPU 사용을 위함
        super().__init__()
        self.device = device
        self.bert = AutoModel.from_pretrained(plm).to(device) #pretrained 모델(plm에 쥐어준 모델) 불러오기
        #drop-out : 0부터 1사이의 확률로 뉴런을 제거하는 기법 _ 특성 feature 만을 과도하게 학습하는 오버피팅을 방지하기 위함
        self.dropout = nn.Dropout(0.1).to(device) # 뉴런별로 0.1의 확률로 drop 될지 여부가 결정됨
        self.classifier = nn.Linear(768, 3).to(device) #BERT의 임베딩 차원은 768임
        self.loss_fct = nn.CrossEntropyLoss().to(device) #MLM을 위한 loss function
        

    def forward(self, batch): # forward : logit 출력을 위한
        #각각 GPU로 조금씩 넘겨줌
        batch["input_ids"] = batch["input_ids"].to(self.device)  #각각의 값들에 접근할 수 있음
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)

        
        outputs = self.bert(**batch) # **batch : dict 형태로 unpacking
        pooled_output = self.dropout(outputs.pooler_output) #dropout 하기
        logits = self.classifier(pooled_output)    # [32,768] -> [32,2]

        return logits