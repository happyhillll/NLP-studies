from constant import START_TOKEN1, START_TOKEN2, END_TOKEN1, END_TOKEN2
from datasets import load_dataset
import numpy as np

dataset = load_dataset("klue", "ner")

def get_character():
    characters = list(set([char for doc in dataset['train'] for char in doc['tokens']]))
    characters.append(START_TOKEN1)
    characters.append(START_TOKEN2)
    characters.append(END_TOKEN1)
    characters.append(END_TOKEN2)

    character={
        char: idx for idx, char in enumerate(characters)
    }
    return character

if __name__ == "__main__":
    get_character()
    
print("a")