from niklueds import HateDataset
from niklugedata import ds
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import lightning as L
from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from nikluemodel import hateClassifierpt

data=ds()

train_data=data.get_train() #data.py에서 데이터 가져와
dev_data=data.get_dev()
tokenizer=AutoTokenizer.from_pretrained("klue/roberta-base") #토크나이저 가져와
train_data=HateDataset(train_data,tokenizer) #ds.py에서 토크나이징해서 데이터셋 만들어
dev_data=HateDataset(dev_data,tokenizer) #ds.py에서 토크나이징해서 데이터셋 만들어
train_loader=DataLoader(train_data, batch_size=20) #데이터 로더 만들어
dev_loader=DataLoader(dev_data, batch_size=20) 

model=hateClassifier()
trainer=L.Trainer(max_epochs=100) #여기에 다양한거 넣어줄 수 있음
trainer.fit(model, train_loader, dev_loader) #데이터 로더 넣어서 학습시켜
    
    # /opt/homebrew/bin/python3 pytorch-lightning