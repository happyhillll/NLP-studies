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
        self.nsp_linear = nn.Linear(768,2).to(device) #NSP를 위한 linear
        self.mlm_linear = nn.Linear(768, 512).to(device) #MLM을 위한 linear
        self.loss_fct = nn.CrossEntropyLoss().to(device) #MLM을 위한 loss function
        self.nsp_loss_fct = LossCls().to(device) #NSP를 위한 loss function
        

    def forward(self, batch): # forward : logit 출력을 위한
        #각각 GPU로 조금씩 넘겨줌
        batch["input_ids"] = batch["input_ids"].to(self.device)  #각각의 값들에 접근할 수 있음
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(self.device)

        
        outputs = self.bert(**batch) # **batch : dict 형태로 unpacking
        pooled_output = self.dropout(outputs.pooler_output) #dropout 하기
        logits = self.classifier(pooled_output)    # [32,768] -> [32,2]

        return logits

<<<<<<< Updated upstream
    # def forward2(self, batch): # forward2 : NSP를 위한 forward
=======
    def forward2(self, batch): # forward2 : MLM과 NSP를 위한 forward
>>>>>>> Stashed changes

    #     batch["input_ids"] = batch["input_ids"].to(self.device)
    #     batch["attention_mask"] = batch["attention_mask"].to(self.device)
    #     batch["token_type_ids"] = batch["token_type_ids"].to(self.device)

<<<<<<< Updated upstream
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
=======
        outputs = self.bert(**batch) # **batch : dict 형태로 unpacking
        hidden = outputs.last_hidden_state  # [32, 512, 768] : 32개 문장, 문장당 512개 토큰, 768차원의 임베딩
        cls_emb = hidden[:,0,:].squeeze()  # [32, 768] :첫번째 attention-head에 대한 feature 출력 의미?
        # self.nsp_linear = nn.Linear(768,2)
        nsp_logits = self.nsp_linear(cls_emb)  # [32, 2]

        mask_index = [[3,7,12], [56,86,123], [543, 768, 434]]  # [32, x] 
        # 첫번쨰 문장에서 3,7,12번째 토큰이 mask되었다는 의미
        # 두번째 문장에서 56,86,123번째 토큰이 mask되었다는 의미
        # 세번째 문장에서 543, 768, 434번째 토큰이 mask되었다는 의미

        for i in range(32):
            cur_mask_index = mask_index[i] #
            mask_emb = hidden[i, cur_mask_index, :]  # [mask 개수, 768] # i번째 문장에서 mask된 토큰들의 임베딩
            # self.mlm_linear = nn.Linear(768,len(vocab))
            self.mlm_linear(mask_emb)  # [mask 개수, vocab 개수] # i번째 문장에서 mask된 토큰들의 임베딩을 vocab 개수만큼의 차원으로 변환


        mlm_labels = hidden[mask_index, :]  # [32, 3, 768]

        return nsp_logits, mlm_labels
>>>>>>> Stashed changes


    # nsp_loss = nsp_loss_fct(nsp_logits, nsp_label)
    # mlm_loss = mlm_loss_fct(mlm_logits, mlm_label)

    # loss = (nsp_loss + mlm_loss) / 2.0
    # loss.backward()