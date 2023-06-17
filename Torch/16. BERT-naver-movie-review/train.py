# train.py
from ds import NSMCDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import BERTClassifier
from data import get_data
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm

"""
References
        https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/bert/modeling_bert.py#L1517
        https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#:~:text=(device)-,Define%20functions%20to%20train%20the%20model%20and%20evaluate%20results.,-import%20time%0A%0Adef
        https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py
"""

DEBUG = True
torch.set_num_threads(4)
device = "cuda:0"

def run():
    n_epoch = 3
    bsz = 32
    lr = 2e-5
    plm = "klue/roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(plm)
    train_data, test_data = get_data("train"), get_data("test")

    if DEBUG:
        train_data, test_data = train_data[:300], test_data[:100]

    train_data, test_data = NSMCDataset(train_data, tokenizer), NSMCDataset(test_data, tokenizer)
    train_loader = DataLoader(train_data, batch_size=bsz)
    test_loader = DataLoader(test_data, batch_size=bsz)
    model = BERTClassifier(plm, device)
    loss_fct = CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.1},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    # Train
    model.train()
    for epoch in range(n_epoch):
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            input, labels = batch
            logits = model(input)  # (bsz, 2)

            labels = labels.to(device)
            loss = loss_fct(logits, labels)
            epoch_loss += loss.detach().float() / bsz
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        print(f"epoch {epoch+1} loss : {epoch_loss}")

        # Evaluation
        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                input, labels = batch
                logits = model(input)       # [32, 2]
                labels = labels.to(device)  # [32]
                total_acc += (logits.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

        print(f"test acc : {total_acc/total_count}")


if __name__ == "__main__":
    run()