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
        
    def preprocess(self,answers,contexts,questions,tokenizer):
        cls_token = tokenizer.cls_token #cls_token : [CLS]
        sep_token = tokenizer.sep_token #sep_token : [SEP]
        unk_token = tokenizer.unk_token #unk_token : [UNK]
        pad_token_id = tokenizer.pad_token_id

    def tokenizing(self,datas,tokenizer):
        tokenizer=AutoTokenizer.from_pretrained("ainize/klue-bert-base-mrc")
        tokenizer(datas['train'][0]['question'],datas['train'][0]['context'],max_length=512,truncation=True,padding="max_length",return_token_type_ids=False)
        print('a')
