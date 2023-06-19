# train.py
from ds import NSMCDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import BERTClassifier
from nsmc_data import *
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm

"""
References
        https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/bert/modeling_bert.py#L1517
        https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#:~:text=(device)-,Define%20functions%20to%20train%20the%20model%20and%20evaluate%20results.,-import%20time%0A%0Adef
        https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py
"""

DEBUG = True
torch.set_num_threads(4)
device = "mps"

def run():
    n_epoch = 3
    bsz = 32
    lr = 2e-5
    plm = "klue/roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(plm)
    train_data, test_data = ds().get_train()[0], ds().get_test()[0]

    # if DEBUG:
    #     train_data, test_data = train_data[:300], test_data[:100]

    train_data, test_data = NSMCDataset(train_data, tokenizer), NSMCDataset(test_data, tokenizer)
    train_loader = DataLoader(train_data, batch_size=bsz)
    test_loader = DataLoader(test_data, batch_size=bsz)
    model = BERTClassifier(plm, device)
    loss_fct = CrossEntropyLoss()

    param_optimizer = list(model.named_parameters()) #모델의 파라미터를 불러옴 : 반복문을 돌때마다 하나씩 뱉어줌 이유는 방식을 다르게 학습 시키려고하는거임
    no_decay = ['bias', "LayerNorm.weight"] #Weight Decay를 적용하지 않을 파라미터를 지정 : 학습이 될수록 미분값이 작아지는데, 이를 방지하기 위해 사용/ 이름에 bias나 layernorm이 들어가는 경우
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], #n : 파라미터 이름, p : 파라미터 값 
         'weight_decay_rate': 0.1},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], #여기서는 bias랑 LayerNorm.weight만 적용
         'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr) 
    
    
if __name__ == '__main__':
    print(run())