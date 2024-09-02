import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self,num_classes,device="mps"):
        super(Net, self).__init__()
        self.device=device
        self.fc1=nn.Linear(28*28, 216).to(device)
        self.fc2=nn.Linear(216,32).to(device)
        self.fc3=nn.Linear(32,16).to(device)
        self.output=nn.Linear(16, num_classes).to(device)
        self.loss_fn=nn.CrossEntropyLoss().to(device)
        self.optimizer=optim.Adam(self.parameters(),lr=0.0001)

    def forward(self, x):
        # (B, 1, 28, 28) -> (B, 28*28)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x=self.output(x)
        return x
    
    def train_model(self,data_loader,num_epoch=20):
        losses=[]
        accs=[]
        
        for epoch in range(num_epoch):
            #loss 초기화
            running_loss = 0
            #정확도 계산
            running_acc = 0
            
            for x,y in data_loader:
                x=x.to(self.device)
                y=y.to(self.device)
                
                #그라디언트 초기화
                self.optimizer.zero_grad()
                
                #output 계산
                output=self(x)
                
                #loss 계산
                loss=self.loss_fn(output,y)
                
                #미분 계산
                loss.backward()
                
                #경사하강법 계산 및 적용
                self.optimizer.step()
                
                # 배치별 loss 를 누적합산 합니다.
                running_loss += loss.item()
                running_acc += output.argmax(dim=1).eq(y).sum().item() / len(y)
            
            # 누적합산된 배치별 loss값을 배치의 개수로 나누어 Epoch당 loss를 산출합니다.
            loss = running_loss / len(data_loader)
            losses.append(loss)
            acc = running_acc / len(data_loader)
            accs.append(acc)

        # 매 Epoch의 학습이 끝날때 훈련 결과를 출력합니다.
        print(f"{epoch:03d} loss = {loss:.5f}, accuracy = {acc:.5f}")
        
        return losses, accs