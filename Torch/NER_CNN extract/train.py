#train.py
import torch.nn as nn
import torch
from dataset import CNNextractDataset
from torch.utils.data import DataLoader
from ds import *
from torch.utils.data import random_split
from model import CNNextract
from vocab import get_character

from datasets import load_dataset
import numpy as np

dataset = load_dataset("klue", "ner")

data = []
for tokens, ner_tags in zip(dataset["train"]["tokens"], dataset["train"]["ner_tags"]):
    sentence_data = [(token, ner_tag) for token, ner_tag in zip(tokens, ner_tags)]
    data.extend(sentence_data)
    
char=get_character() #char 사전
dataset = CNNextractDataset(data, char)
loader=DataLoader(dataset,batch_size=32)
model=CNNextract() 

for i,batch in enumerate(loader):
    x,y=batch
    y_hat=model(x)