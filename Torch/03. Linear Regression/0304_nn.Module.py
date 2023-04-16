#단순선형회귀
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#데이터
x_train=torch.FloatTensor([[1],[2],[3]])
y_train=torch.FloatTensor([[2],[4],[6]])

#모델을 선언 및 초기화
model = nn.Linear(1,1) #입력과 출력의 차원을 각각 1로 설정
#model = torch.nn.Linear(1,1) #위와 동일한 코드

#model에 가중치와 편향이 저장되어져 있음
print(list(model.parameters()))

#optimizer 설정. 경사하강법 SGD를 사용하고, 학습률은 0.01
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#전체 훈련 데이터에 대해 경사하강법을 2000번 수행
nb_epochs=2000
for epoch in range(nb_epochs+1):
    #H(x) 계산
    prediction=model(x_train)
    
    #cost 계산
    cost=F.mse_loss(prediction,y_train)
    
    #cost로 H(x) 개선
    #gradient를 0으로 초기화
    optimizer.zero_grad()
    #비용함수를 미분하여 gradient 계산
    cost.backward()
    #W와 b를 업데이트
    optimizer.step()
    
    if epoch % 100==0:
        #100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch,nb_epochs,cost.item()))

print(list(model.parameters()))

#x에 임의의 값 4를 넣어서 예측값을 출력
new_var=torch.FloatTensor([[4.0]])
pred_y=model(new_var)
print("훈련 후 입력이 4일 때의 예측값 : ",pred_y)

#다중 선형 회귀 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

x_train=torch.FloatTensor([[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]])
y_train=torch.FloatTensor([[152],[185],[180],[196],[142]])

model=nn.Linear(3,1)
print(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    #H(x) 계산
    prediction = model(x_train)
    #model(x_train)은 내부적으로 model.forward(x_train)을 호출
    
    #cost 계산
    cost=F.mse_loss(prediction,y_train)
    
    #cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch%100==0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch,nb_epochs,cost.item()))
        
print(list(model.parameters()))

#x에 입의의 값 73,80,75를 넣어서 예측값을 출력
new_var=torch.FloatTensor([[73,80,75]])
y_pred=model(new_var)
print("훈련 후 입력이 73,80,75일 때의 예측값 : ",y_pred)