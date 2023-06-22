# train.py

from ds import NSMCDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import *
from data import get_data
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
from ds import DialogDataset


"""
순서
    0. 논문 읽기
        abstract (읽을만한가, 나한테 필요한건가)
        intro (읽을만한가, 나한테 필요한건가)
        결론 (읽을만한가, 나한테 필요한건가)
        방법론 그림 -> 아 읽자
        다 읽기
    1. 모델부분 망상
    2. 모델에서 필요한 dataset 구조 결정
        BERT -> 1 doc -> n segment -> BERT로 한번에 들어가면 되겠다. (n, 512) -> (n,512,768)
    3. get_data
    4. Dataset으로 변형
    5. 모델 구현
    6. train 완성
"""

"""
https://arxiv.org/pdf/1910.10781.pdf
"""


DEBUG = True
torch.set_num_threads(4)
device = "cuda:0"

def run():
    n_epoch = 3
    bsz = 1
    lr = 2e-5
    plm = "klue/roberta-base"


    datas = get_data()
    train_data, test_data = datas[:int(len(datas)*0.9)], datas[int(len(datas)*0.9):]

    tokenizer = AutoTokenizer.from_pretrained(plm)
    train_data, test_data = DialogDataset(train_data, tokenizer), DialogDataset(test_data, tokenizer)
    train_loader = DataLoader(train_data, batch_size=bsz)
    test_loader = DataLoader(test_data, batch_size=bsz)
    model = LongDocClassifier(plm, device)
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
    for epoch in range(n_epoch):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            input, labels = batch
            input['input_ids'] = input['input_ids'].squeeze()
            input['attention_mask'] = input['attention_mask'].squeeze()
            input['token_type_ids'] = input['token_type_ids'].squeeze()

            logits = model(input)

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
                input['input_ids'] = input['input_ids'].squeeze()
                input['attention_mask'] = input['attention_mask'].squeeze()
                input['token_type_ids'] = input['token_type_ids'].squeeze()
                logits = model(input)       # [32, 2]
                labels = labels.to(device)  # [32]
                total_acc += (logits.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

        print(f"test acc : {total_acc/total_count}")


if __name__ == "__main__":
    run()