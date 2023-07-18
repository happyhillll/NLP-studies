from QAdata import *
from QAds import *
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch
import lightning as L
from torch.optim.optimizer import Optimizer
from transformers import AutoModel, AutoTokenizer

'''
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ainize/klue-bert-base-mrc")
model = AutoModelForQuestionAnswering.from_pretrained("ainize/klue-bert-base-mrc")

context = "your context"
question = "your question"

encodings = tokenizer(context, question, max_length=512, truncation=True,
                      padding="max_length", return_token_type_ids=False)
encodings = {key: torch.tensor([val]) for key, val in encodings.items()}             

input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]

pred = model(input_ids, attention_mask=attention_mask)

start_logits, end_logits = pred.start_logits, pred.end_logits

token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)

pred_ids = input_ids[0][token_start_index: token_end_index + 1]

prediction = tokenizer.decode(pred_ids)
'''

dss=ds()

train_data = dss.get_train()
tokenizer=AutoTokenizer.from_pretrained("ainize/klue-bert-base-mrc")
train_data = QADataset(train_data[2],train_data[1],train_data[0],tokenizer)
print('a')
