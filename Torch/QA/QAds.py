from torch.utils.data import Dataset
import torch
import random
from QAdata import *
from transformers import AutoTokenizer
    
# https://ainize.ai/workspace/view?ipynb=https://raw.githubusercontent.com/ainize-team/klue-mrc-workspace/master/klue-bert-base-mrc.ipynb&imageId=HQ8gBR4qbSwEEgcoJL4G&utm_medium=social&utm_source=medium&utm_campaign=everyone%20ai&utm_content=klue
#https://github.com/huggingface/transformers/blob/abaca9f9432a84cfaa95531de4c72334f38a42f2/src/transformers/models/bert/modeling_bert.py#L1797
#https://github.com/LostCow/KLUE/blob/main/mrc/processor.py#L80

class QADataset(Dataset):
    def __init__(self,answers,contexts,data,tokenizer):
        self.preprocess(data,tokenizer)
        self.add_end_idx(answers,contexts)
        self.tokenizing(data,tokenizer)
        
    def preprocess(self,datas,tokenizer):
        question = ds().get_train()[0]
        context = ds().get_train()[1]
        answer=ds().get_train()[2]
    
    def add_end_idx(self,answers,contexts):
        for answer, context in zip(answers, contexts):
            real_answer=answer['text'] #real_answer=['한 달가량','한 달']
            start_idx=answer['answer_start'] #start_idx=[478,478]
            for w in real_answer: #w='한 달가량'
                end_idx=[]
                for i in start_idx:
                    end_idx.append(str(i+len(w)))
                    answer['answer_end']=end_idx
            
            
            #이거 꼭 해줘야하는지? : 하려고 하니까 리스트에 2개가 들어와있는 경우 for문을 다시 돌려야 하는거 같음
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
        
        return answers,contexts    

        # 토크나이징 하고
        # train+question 붙이고
        # 학습을 그냥 두번 하면 되는지????
    def tokenizing(self,datas,tokenizer):
        tokenizer=AutoTokenizer.from_pretrained("ainize/klue-bert-base-mrc")
        tokenizer(datas['train'][0]['question'],datas['train'][0]['context'],max_length=512,truncation=True,padding="max_length",return_token_type_ids=False)
        print('a')