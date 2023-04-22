def train():
    loader = get_data_loader()
    model=Classifier(28*28, 10)
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
    nb_epochs=20
    criterion=nn.CrossEntropyLoss()
    for epoch in range(nb_epochs+1):
        epoch_loss=0
        for batch_idx, batch in enumerate(loader):
            n_sample=batch['image'].shape[0]
            image = batch['image'].view(n_sample, -1) #[n_sample, 28, 28]
            preds = model(image) # [n_sample, 28*28] -> [n_sample, 10]
            
            loss = criterion(preds, batch['label'])
            epoch_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('\tEpoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(
                    epoch, nb_epochs, batch_idx + 1, len(loader),
                    loss.item()
                ))

            print('Epoch {:4d}/{} Epoch Loss: {:.6f}'.format(
                epoch, nb_epochs, epoch_loss
            ))

if __name__=="__main__":
    train()
    

#그럼 함수 이름이 절대 겹치면 안되는건지?