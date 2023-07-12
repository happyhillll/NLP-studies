# data.py

from datasets import load_dataset

def get_qa_data():
    dataset = load_dataset('klue', 'mrc')
    return dataset['train'], dataset['validation']