# main > train, eval / 아니면 따로 놔도 됨
# model
# data 관련

from typing import Any, Callable, Optional, Union
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch
import lightning as L
from torch.optim.optimizer import Optimizer
from transformers import AutoModel
from hatedata import *
from torch.utils.data import DataLoader
from model import hateClassifier
import torch
print(torch.cuda.is_available())
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dss=ds()

train_data = dss.get_train()
dev_data = dss.get_test()
train_loader = DataLoader(train_data, batch_size=5)
dev_loader=DataLoader(dev_data, batch_size=5)

model=hateClassifier()
trainer=L.Trainer(max_epochs=-1)
trainer.fit(model, train_loader, dev_loader)