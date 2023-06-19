# model.py
# cuda 설치하기

import torch.nn as nn
import torch
from transformers import AutoModel

class BERTClassifier(nn.Module):
    def __init__(self, plm, device): #plm : pretrained language model, device : GPU 사용을 위함
        super().__init__()
        self.device = device
        self.bert = AutoModel.from_pretrained(plm).to(device) #pretrained 모델(plm에 쥐어준 모델) 불러오기
        #drop-out : 0부터 1사이의 확률로 뉴런을 제거하는 기법 _ 특성 feature 만을 과도하게 학습하는 오버피팅을 방지하기 위함
        self.dropout = nn.Dropout(0.1).to(device) # 뉴런별로 0.1의 확률로 drop 될지 여부가 결정됨
        self.classifier = nn.Linear(768, 2).to(device) #BERT의 임베딩 차원은 768임


    def forward(self, batch): # forward : MLM을 위한 forward
        #각각 GPU로 조금씩 넘겨줌
        batch["input_ids"] = batch["input_ids"].to(self.device)  #각각의 값들에 접근할 수 있음
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)

        
        outputs = self.bert(**batch) # **batch : dict 형태로 unpacking
        pooled_output = self.dropout(outputs.pooler_output) # ?
        logits = self.classifier(pooled_output)    # [32,768] -> [32,2]

        return logits

    # def forward2(self, batch): # forward2 : NSP를 위한 forward

    #     batch["input_ids"] = batch["input_ids"].to(self.device)
    #     batch["attention_mask"] = batch["attention_mask"].to(self.device)
    #     batch["token_type_ids"] = batch["token_type_ids"].to(self.device)

    #     outputs = self.bert(**batch)
    #     hidden = outputs.last_hidden_state  # [32, 512, 768]
    #     cls_emb = hidden[:,0,:].squeeze()  # [32, 768]
    #     # self.nsp_linear = nn.Linear(768,2)
    #     nsp_logit = self.nsp_linear(cls_emb)  # [32, 2]

    #     mask_index = [[3,7,12], [56,86,123], [543, 768, 434]]  # [32, x]

    #     for i in range(32):
    #         cur_mask_index = mask_index[i]
    #         mask_emb = hidden[i, cur_mask_index, :]  # [mask 개수, 768]
    #         # self.mlm_linear = nn.Linear(768,len(vocab))
    #         self.mlm_linear(mask_emb)  # [mask 개수, vocab 개수]


    #     mask_emb = hidden[mask_index, :]  # [32, 3, 768]

    #     return nsp_logits, mlm_lgots


    # nsp_loss = nsp_loss_fct(nsp_logits, nsp_label)
    # mlm_loss = mlm_loss_fct(mlm_logits, mlm_label)

    # loss = (nsp_loss + mlm_loss) / 2.0
    # loss.backward()