from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch
import lightning as L
from torch.optim.optimizer import Optimizer
from transformers import AutoModel, AutoTokenizer
from bertNERdata import *
from NERds import *
from torch.utils.data import DataLoader
# from model import NERClassifier

dss=ds()

train_data = dss.get_train()
dev_data = dss.get_val()
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
train_data=NERDataset(train_data,tokenizer)
train_loader = DataLoader(train_data, batch_size=5)
dev_loader=DataLoader(dev_data, batch_size=5)

model=NERClassifier()
trainer=L.Trainer(max_epochs=-1)
trainer.fit(model, train_loader, dev_loader)