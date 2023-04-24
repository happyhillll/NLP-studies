'''
MNIST에서 32 mini batch는 
32개의 숫자를 의미함.. 그러면 nlp에서도 32 mini batch이면 32개의 문장과 같은 것인가?

'''

def train():
    loader=get_data_loader() #loader는 반복 가능한 object
    model=Fashion_MNIST_Classifier(28*28,10) #input size : 28*28, output size : 10
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-5) 
    nb_epochs=20
    criterion=nn.CrossEntropyLoss() #loss function
    for epoch in range(nb_epochs+1): 
        epoch_loss=0
        for batch_idx, batch in enumerate(loader): #loader에서 batch를 하나씩 가져온다.
            n_sample=batch['image'].shape[0] #현재 batch에 있는 image의 개수(32가 아닐 수도 있으므로)
            image=batch['image'].view(n_sample,-1) #batch['image']하나 당, 1차원 벡터로 변환 784개의 원소를 가진 벡터
            preds=model(image) # [n_sample, 28*28] -> [n_sample, 10]
            
            loss=criterion(preds,batch['label']) #loss 계산
            epoch_loss+=loss #epoch_loss에 loss를 더해준다.
            optimizer.zero_grad() #optimizer의 gradient를 0으로 초기화
            loss.backward() #loss를 back propagation
            #backward란? https://www.youtube.com/watch?v=Ilg3gGewQ5U
            optimizer.step() #optimizer의 step을 실행
            
            print('\tEpoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
                    epoch, nb_epochs, batch_idx + 1, len(loader),
                    loss.item()
                ))

            print('Epoch {:4d}/{} Epoch Loss: {:.6f}'.format(
                epoch, nb_epochs, epoch_loss
            ))
            
if __name__=="__main__":
    train()

#파라미터 체크
print(list(model.parameters()))

