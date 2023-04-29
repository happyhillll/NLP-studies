from leaf_data import get_data_loader
from classifier import Classifier
import numpy as np
import torch
import torch.nn as nn

#train.py를 실행하면 자동으로 leaf_data가 실행되는데 1행 때문인지?
#leaf_data 디버깅해서 shape 확인하고 싶음

def train():
    train_loader =get_data_loader()
    model=Classifier(64*64, 33)
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-5)
    nb_epochs=10
    criterion=nn.CrossEntropyLoss()
    for epoch in range(nb_epochs+1):
        epoch_loss=0
        for batch_idx, (data, target) in enumerate(train_loader): #(data,target) 형태로 묶여져 있음
            n_sample=data.shape[0]
            image=data.view(n_sample,-1)
            preds=model(image)
            
            #loss
            loss=criterion(preds,target)
            epoch_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
            print('\tEpoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
                    epoch, nb_epochs, batch_idx + 1, len(train_loader),
                    loss.item()
                ))

            print('Epoch {:4d}/{} Epoch Loss: {:.6f}'.format(
                epoch, nb_epochs, epoch_loss
            ))

if __name__=="__main__":
    train()