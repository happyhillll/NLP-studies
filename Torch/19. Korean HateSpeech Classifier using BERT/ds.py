# train.py
from pprint import pprint
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from data import get_data
from ds import NSMCDataset
import yaml
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import datetime as dt
from model import Classifier
from transformers import AutoTokenizer

DEBUG = True
# DEBUG = False

torch.set_num_threads(8)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train():
    n_epoch = 5
    bsz = 32
    lr = 2e-5
    n_workers = 6
    plm = "klue/roberta-base"
    dir = "./ckpt/"
    ckpt = 'model-{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}'
    monitor = 'val_acc'

    train_data, test_data = get_data("train"), get_data("test")

    if DEBUG:
        train_data, val_data, test_data = train_data[:300], train_data[300:400], test_data[:100]
    else:
        train_data, val_data = train_data[:int(len(train_data) * 0.9)], train_data[int(len(train_data) * 0.9):]

    tokenizer = AutoTokenizer.from_pretrained(plm)

    train_data, val_data, test_data = NSMCDataset(train_data, tokenizer), NSMCDataset(val_data, tokenizer), NSMCDataset(test_data, tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=bsz, num_workers=n_workers)
    val_dataloader = DataLoader(val_data, batch_size=bsz, num_workers=n_workers)
    test_dataloader = DataLoader(test_data, batch_size=bsz, num_workers=n_workers)

    model = Classifier(plm, tokenizer)

    model.set_configuration(epochs=n_epoch, batch_size=bsz, lr=lr, data_len=len(train_data))

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        dirpath=dir,
        filename=ckpt,
        save_top_k=-1,
        )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=n_epoch,
        num_sanity_val_steps=-1,
        callbacks=[checkpoint_callback, ],
        check_val_every_n_epoch=1,
        )


    trainer.fit(model, train_dataloader, val_dataloader)


    ckpt_path = Path(dir)
    best_model_path = ckpt_path / Path(checkpoint_callback.best_model_path).parts[-1]
    best_model_writer = ckpt_path / 'best.ckpt'
    with open(best_model_writer, 'w', encoding='utf-8-sig') as f:
        f.write(str(best_model_path))

    model.freeze()

    # trainer.test(model, dataloaders=test_dataloader)
    trainer.test(
        ckpt_path='best',
        dataloaders=test_dataloader,
        )
    print(len(test_dataloader))
    # writejson(model.result, "temp.json")


if __name__ == '__main__':
    train()