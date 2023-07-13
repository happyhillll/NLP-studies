from torch.utils.data import Dataset
import torch
import random
from QAdata import *
    
# https://ainize.ai/workspace/view?ipynb=https://raw.githubusercontent.com/ainize-team/klue-mrc-workspace/master/klue-bert-base-mrc.ipynb&imageId=HQ8gBR4qbSwEEgcoJL4G&utm_medium=social&utm_source=medium&utm_campaign=everyone%20ai&utm_content=klue
#https://github.com/huggingface/transformers/blob/abaca9f9432a84cfaa95531de4c72334f38a42f2/src/transformers/models/bert/modeling_bert.py#L1797
#https://github.com/LostCow/KLUE/blob/main/mrc/processor.py#L80

class QADataset(Dataset):
    def __init__(self,answers,contexts,data,tokenizer):
        self.preprocess(data,tokenizer)
        self.add_end_idx(answers,contexts)
        
    def preprocess(self,datas,tokenizer):
        question = ds().get_train()[0]
        context = ds().get_train()[1]
        answer=ds().get_train()[2]
    
    def add_end_idx(self,answers,contexts):
        for answer, context in zip(answers, contexts):
            real_answer=answer['text']
            start_idx=answer['answer_start']
            # end_idx=start_idx+