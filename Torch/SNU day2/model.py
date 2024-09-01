import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_features, device="mps"):
        super(Net, self).__init__()
        self.device=device
        self.fc1=nn.Linear(num_features, 32).to(device)
        self.fc2=nn.Linear(32,16).to(device)
        self.fc3=nn.Linear(16,8).to(device)
        self.output=nn.Linear(8,1).to(device)
        self.loss_fn=nn.MSELoss().to(device)
        self.optimizer=optim.Adam(self.parameters(),lr=0.0001)
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.output(x)
        return x
    
    def train_model(self,data_loader,num_epoch=200):
        losses=[]
        
        for epoch in range(num_epoch):
            #loss 초기화
            running_loss=0.0
            for x,y in data_loader:
                x=x.to(self.device)
                y=y.to(self.device)
                
                #그라디언트 초기화
                self.optimizer.zero_grad()
                
                #output 계산 : forward
                y_hat=self(x)
                
                #loss 계산
                loss=self.loss_fn(y.view(-1,1), y_hat)
                
                #미분 계산
                loss.backward()
                
                #경사하강법 업데이트
                self.optimizer.step()
                
                #배치별 lossfmf 누적합산
                running_loss += loss.item()
                
            #평균 loss 계산
            loss=running_loss / len(data_loader)
            losses.append(loss)
            
            #20번의 Epoch당 출력
            if epoch%20 == 0:
                print("{0:05d} loss = {1:.5f}".format(epoch, loss))
        
        print("----" * 15)
        print("{0:05d} loss = {1:.5f}".format(num_epoch - 1, loss))

        return losses       