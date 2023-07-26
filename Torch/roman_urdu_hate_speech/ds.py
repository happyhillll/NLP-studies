from torch.utils.data import Dataset
import torch
import random
from Romandata import *
from transformers import AutoTokenizer

class RomanDataset(Dataset):
    def __init__(self,datas,tokenizer):
        self.preprocess(datas,tokenizer)