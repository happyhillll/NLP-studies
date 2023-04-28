# data.py
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import torch

def get_data_loader():
    train_ds, test_ds = load_dataset("mnist", split=["train", "test"])
    def transform_func(examples):
        examples["image"] = [torch.FloatTensor(np.array(img)) for img in examples["image"]]
        return examples
    train_ds = train_ds.with_transform(transform_func)
    test_ds = test_ds.with_transform(transform_func)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)
    return train_loader, test_loader

'''
디버깅 질문
1. def 안의 def 안은 어떻게 접근할 수 있는지??
2. train_ds의 shape을 확인하고 싶은데 어디에서 확인할 수 있는건지 모르겠음.
'''

print("data.py")
if __name__ == "__main__":
    get_data_loader()

#classifier
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

# train.py
# from classfier import Classifier
# from data import get_data_loader
import torch
import torch.nn as nn


def train():
    train_loader, test_loader = get_data_loader()
    model = Classifier(28*28, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    nb_epochs = 10
    criterion = nn.CrossEntropyLoss()
    for epoch in range(nb_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # if batch_idx == 10:
            #     break

            n_sample = batch['image'].shape[0]
            image = batch['image'].view(n_sample, -1)  # [n_sample, 28, 28] -> [n_sample, 28*28]
            preds = model(image)  # [n_sample, 28*28] -> [n_sample, 10]

            loss = criterion(preds, batch['label'])
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('\tEpoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
            #     epoch, nb_epochs, batch_idx + 1, len(train_loader),
            #     loss.item()
            # ))

        print('Epoch {:4d}/{} Epoch Train Loss: {:.6f}'.format(
            epoch + 1, nb_epochs, epoch_loss
        ))

        test_preds, test_labels = [], []
        for batch_idx, batch in enumerate(test_loader):
            # if batch_idx == 10:
            #     break
            n_sample = batch['image'].shape[0]
            image = batch['image'].view(n_sample, -1)  # [n_sample, 28, 28] -> [n_sample, 28*28]
            preds = model(image)  # [n_sample, 28*28] -> [n_sample, 10]

            # Q1. 왜 loss를 구하지 않을까용?
            preds = torch.argmax(preds, dim=1)  # [n_sample, 10] -> [n_sample,]
            # Q2. 왜 softmax를 쓰지 않죠?
            test_preds.append(preds)
            test_labels.append(batch['label'])

        # test_preds, test_labels = map(torch.concat, [test_preds, test_labels])
        test_preds = torch.concat(test_preds)
        test_labels = torch.concat(test_labels)
        tp = torch.sum(test_labels == test_preds)
        acc = tp.item() / test_preds.numel()
        print(f"Epoch {epoch + 1:4d}/{nb_epochs} Epoch Test Acc.: {acc:.6f}\n")


if __name__ == "__main__":
    train()