from torch.utils.data import Dataset
import torch
import random
from QAdata import *
from transformers import AutoTokenizer
    
# https://ainize.ai/workspace/view?ipynb=https://raw.githubusercontent.com/ainize-team/klue-mrc-workspace/master/klue-bert-base-mrc.ipynb&imageId=HQ8gBR4qbSwEEgcoJL4G&utm_medium=social&utm_source=medium&utm_campaign=everyone%20ai&utm_content=klue
#https://github.com/huggingface/transformers/blob/abaca9f9432a84cfaa95531de4c72334f38a42f2/src/transformers/models/bert/modeling_bert.py#L1797
#https://github.com/LostCow/KLUE/blob/main/mrc/processor.py#L80

class QADataset(Dataset):
    def __init__(self,answers,contexts,questions,tokenizer):
        self.preprocess(answers,contexts,questions,tokenizer)
        self.add_end_idx(answers,contexts)
            
<<<<<<< Updated upstream
            # #이거 꼭 해줘야하는지? : 하려고 하니까 리스트에 2개가 들어와있는 경우 for문을 다시 돌려야 하는거 같음
            # for start_id,end_id in zip(start_idx,end_idx):
            #     start_id=int(start_id)
            #     end_id=int(end_id)
            #     if context[start_id:end_id]==w:
            #         answer['answer_end']=end_idx
            #     elif context[start_id-1:end_id-1]==w:
            #         answer['answer_start']=start_id-1
            #         answer['answer_end']=end_id-1
            #     elif context[start_id-2:end_id-2]==w:
            #         answer['answer_start']=start_id-2
            #         answer['answer_end']=end_id-2
=======
        self.tokenizer=tokenizer
        self.answers=answers
        self.contexts=contexts
        self.model_max_position_embeddings=512
        self.encodings=self.tokenizer(self.contexts, self.questions,max_length=512, truncation=True, padding="max_length",return_token_type_ids=False)
        self.add_token_positions()
>>>>>>> Stashed changes
        
        
    # def preprocess(self,answers,contexts,questions,tokenizer):
    #     cls_token = tokenizer.cls_token #cls_token : [CLS]
    #     sep_token = tokenizer.sep_token #sep_token : [SEP]
    #     unk_token = tokenizer.unk_token #unk_token : [UNK]
    #     pad_token_id = tokenizer.pad_token_id

<<<<<<< Updated upstream
    def tokenizing(self,datas,tokenizer):
        tokenizer=AutoTokenizer.from_pretrained("ainize/klue-bert-base-mrc")
        tokenizer(datas['train'][0]['question'],datas['train'][0]['context'],max_length=512,truncation=True,padding="max_length",return_token_type_ids=False)
        print('a')
=======
    #     featrues=[]
    #     unk_token=-100
    
    
    # answer의 시작, 종료 위치를 토큰화된 문장에서 찾아서 context안에서 몇번째 토큰인지 찾는다.
    # 해야할것 1. 원래 문자열에서 토큰의 인덱스 가져오기
    def add_token_positions(self):
        start_positions=[]
        end_positions=[]
        for i in range(len(self.answers)):
            start_positions.append(self.encodings.char_to_token(i, self.answers[i]['answer_start']))
            end_positions.append(self.encodings.char_to_token(i, self.answers[i]['answer_end'] - 1))
            
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        
        self.encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        
    def get_encodings(self):
        return self.encodings
        # 토크나이징 하고
        # train+question 붙이고
        # 학습을 그냥 두번 하면 되는지????

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
>>>>>>> Stashed changes
