#1. 단순 선형 회귀 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#데이터 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#모델 초기화
model= nn.Linear(1, 1) #입력과 출력의 차원을 지정 : 1차원 입력, 1차원 출력(하나의 입력 x에 대해 하나의 출력 y를 가짐)

print(list(model.parameters())) #가중치 W와 편향 b 랜덤 초기화

#옵티마이저 정의 : 경사 하강법 SGD 사용
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#epochs
nb_epochs = 2000
for epoch in range(nb_epochs+1):
    #H(x) 계산
    prediction = model(x_train)
    #cost 계산
    cost= F.mse_loss(prediction, y_train)
    #cost로 H(x) 개선
    #gradient를 0으로 초기화
    optimizer.zero_grad()
    #cost를 미분하여 gradient 계산
    cost.backward()
    #W와 b를 업데이트
    optimizer.step()
    
    if epoch % 100 == 0:
        #100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
        
#임의의 값4를 넣어 W와 b의 값이 최적화가 되었는지 확인하자
new_var = torch.FloatTensor([[4.0]])
#입력한 값에 대한 예측값을 출력
pred_y=model(new_var) #forward 연산 : 입력값을 넣어서 출력값을 얻는 것
#y=2x이므로 4x2=8이 나와야 함
print("훈련 후 입력이 4일 때의 예측값 : ", pred_y)

print(list(model.parameters())) #학습된 W와 b 출력
