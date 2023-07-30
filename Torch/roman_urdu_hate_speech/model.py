# model.py
from pprint import pprint
import pytorch_lightning as pl
from typing import Optional
import torch.nn as nn
import torch
from torch.nn import functional as F
from collections import defaultdict
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import BinaryAccuracy, Accuracy

from transformers import (
    PreTrainedModel,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertModel,
    RobertaModel,
    BertConfig,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Classifier(pl.LightningModule):
    """
    https://aclanthology.org/2020.emnlp-main.197.pdf
    https://wikidocs.net/63618
    https://blog.joonas.io/196
    """
    def __init__(
            self,
            model_name_or_path: str,
            tokenizer,
            dropout_rate: float = 0.1,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path

        self.tokenizer = tokenizer

        self.bert = AutoModel.from_pretrained(model_name_or_path)

        # self.cnn1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2))


        self.linear = nn.Linear(768, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=2)  # BinaryAccuracy(multidim_average='samplewise')

    def compute_loss(self, tokens, labels):
        """
        https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/bert/modeling_bert.py#L1517
        """
        out = self.bert(**tokens)
        pooled_output = self.dropout(out.pooler_output)     # [bsz, dem]
        logits = self.linear(pooled_output)                 # [bsz, 2]
        loss = self.criterion(logits.squeeze(), labels.squeeze())
        return loss, logits

    def training_step(self, batch, batch_idx):  # batch == 1 doc
        tokens, labels = batch
        loss, _ = self.compute_loss(tokens, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        loss, logits = self.compute_loss(tokens, labels)
        preds = logits.argmax(1)
        self.acc.update(logits, labels)

        metrics = {"loss": loss, "preds": preds, "labels": labels}
        return metrics

    def validation_epoch_end(self, outputs):
        acc = self.acc.compute()
        self.acc.reset()

        outputs = self.all_gather(outputs)
        if self.trainer.is_global_zero:
            val_losses = [output["loss"].mean() for output in outputs]
            val_loss_mean = sum(val_losses) / len(val_losses)
            self.log("val_loss", val_loss_mean)     # for ckpt file name
            self.log("val_acc", acc)                # for ckpt file name

            print("Val. Acc")
            print(acc)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        acc = self.acc.compute()
        self.acc.reset()

        outputs = self.all_gather(outputs)
        if self.trainer.is_global_zero:
            val_losses = [output["loss"].mean() for output in outputs]
            val_loss_mean = sum(val_losses) / len(val_losses)
            self.log("test_loss", val_loss_mean)    # for ckpt file name
            self.log("test_acc", acc)               # for ckpt file name

            print("Test. Acc")
            print(acc)

            self.result = outputs

    def predict_step(self, batch, batch_idx):
        pass

    def inference(self):
        pass

    def set_configuration(
            self,
            epochs=5,
            batch_size=1,
            lr: float = 2e-5,
            data_len: int = 1000,
            warmup_ratio: float = 0.0,
            epsilon: float = 1e-8,
            weight_decay: float = 0.0
    ):

        self.epochs = epochs
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.data_len = data_len

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.epsilon)

        # warm up lr

        num_train_steps = self.data_len * self.epochs
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        lr_scheduler = {
            'scheduler': scheduler,
            'monitor': 'loss',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]