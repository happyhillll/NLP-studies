'''
class 개수로 차원을 바꿔

forward : 
'''

model=nn.Linear(1,1)

class LinearRegressionModel(nn.Module): #nn.Module을 상속받음
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,1)
    
    def forward(self,x):
        return self.linear(x)

#단순 선형 회귀 클래스로 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#데이터
x_train=torch.FloatTensor([[1],[2],[3]])
y_train=torch.FloatTensor([[2],[4],[6]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    
    def forward(self,x):
        return self.linear(x)
    
    
model=LinearRegressionModel()

#optimizer 설정
optimizer=optim.SGD(model.parameters(), lr=0.01)

#전체 훈련 데이터에 대해 경사 하강법을 2000번 수행
nb_epochs=2000
for epoch in range(nb_epochs+1):
    #H(x) 계산
    hypothesis=model(x_train)
    #cost 계산
    cost=F.mse_loss(hypothesis, y_train)
    #cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 ==0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

#다중 선형 회귀 클래스로 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#데이터
x_train=torch.FloatTensor([[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]])
y_train=torch.FloatTensor([[152],[185],[180],[196],[142]])

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,1)
    
    def forward(self,x):
        return self.linear(x)
    
model=MultivariateLinearRegressionModel()
optimizer=optim.SGD(model.parameters(), lr=1e-5)

nb_epochs=2000
for epoch in range(nb_epochs+1):
    #H(x) 계산
    hypothesis=model(x_train)
    #cost 계산
    cost=F.mse_loss(hypothesis, y_train)
    #cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 ==0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
        
        