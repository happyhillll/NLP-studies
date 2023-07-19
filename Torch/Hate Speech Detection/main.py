from ds import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import lightning as L
from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from model import hateClassifier
from niklugedata import ds

data=ds()

train_data=data.get_train() #data.py에서 데이터 가져와
dev_data=data.get_dev()
tokenizer=AutoTokenizer.from_pretrained("klue/roberta-base") #토크나이저 가져와
train_data=HateDataset(train_data,tokenizer) #ds.py에서 토크나이징해서 데이터셋 만들어
train_loader=DataLoader(train_data, batch_size=1) #데이터 로더 만들어
dev_loader=DataLoader(dev_data, batch_size=1) 

model=hateClassifier()
trainer=L.Trainer(max_epochs=1000)
trainer.fit(model, train_loader, dev_loader) #데이터 로더 넣어서 학습시켜