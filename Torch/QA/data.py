# data.py
# from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset

class ds:
    def get_qa_data():
        dataset = load_dataset('klue', 'mrc')
        return dataset['train'], dataset['validation']