from ds import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
import lightning as L
from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from model import NER_classifier
from nerdata import *

data=ds()

train_data = data.get_train()
dev_data = data.get_val()
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
train_data=dataset(train_data,tokenizer)
train_loader = DataLoader(train_data, batch_size=5)
dev_loader=DataLoader(dev_data, batch_size=5)

model=NER_classifier()
trainer=L.Trainer(max_epochs=-1)
trainer.fit(model, train_loader, dev_loader)
