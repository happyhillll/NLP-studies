'''
train/valid/test
8:1:1

tp : true positive

loader는 반복 가능한 object

NSP : Next Sentence Prediction

'''
from datasets import load_dataset
from torch.utils.data import DataLoader  #SGD의 반복 연산을 실행할때 사용하는 미니 배치용 유틸리티 함수
from MNISTdata import get_data_loader
from classifier import Classifier
import numpy as np
import torch

def train():
    train_loader, test_loader = get_data_loader()
    model=Classifier(28*28, 10) #input size : 28*28, output size : 10
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-5) 
    nb_epochs=20
    criterion=nn.CrossEntropyLoss()
    for epoch in range(nb_epochs+1):
        epoch_loss=0
        for batch_idx, batch in enumerate(train_loader):
            n_sample=batch['image'].shape[0] #현재 batch의 sample 수
            image = batch['image'].view(n_sample, -1) #[n_sample, 28, 28]
            preds = model(image) # [n_sample, 28*28] -> [n_sample, 10]
            
            #loss
            loss = criterion(preds, batch['label'])
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

# if __name__=="__main__":
#     train()

#파라미터 체크
# print(list(model.parameters()))

#test
test_preds, test_labels = [], []
for batch_idx, batch in enumerate(test_loader):
    n_sample=batch['image'].shape[0] #현재 batch의 sample 수
    image=batch['image'].view(n_sample, -1) # [n_sample, 28, 28] -> [n_sample, 28*28]
    preds=model(image) # [n_sample, 28*28] -> [n_sample, 10]
    
    #왜 loss를 구하지 않을까? : test니까
    
    #열 별로 argmax구하기
    preds=torch.argmax(preds, dim=1) # [n_sample, 10] -> [n_sample]
    #왜 softmax를 쓰지 않을까? argmax는 최대값을 가진 인덱스를 리턴해주니까,굳이 쓰지 않아도 됨.
    test_preds.append(preds) # 예측한 값을 넣어주고
    test_labels.append(batch['label']) # 실제 label을 넣어준다.
    
    test_preds=torch.concat(test_preds) #일자로 펴준다.
    test_labels=torch.concat(test_labels)
    tp = torch.sum(test_labels==test_preds) #실제 label과 예측한 label이 같은 것들의 개수
    acc = tp/len(test_labels)
    print(f"Epoch {epoch + 1:4d}/{nb_epochs} Epoch Test Acc.: {acc:.6f}\n")

if __name__ =="__main__":
    train()

