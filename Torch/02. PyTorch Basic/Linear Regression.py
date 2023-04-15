import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) # 랜덤 시드 고정 (난수 발생 순서와 값을 동일하게 보장해줌)

# 변수 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)

#가중치 W를 0으로 초기화하고, 학습을 통해 값이 변경되도록 설정
W=torch.zeros(1, requires_grad=True) #requires_grad=True : 자동 미분 기능 적용
print(W)
#편향 b를 0으로 초기화하고, 학습을 통해 값이 변경되도록 설정
b=torch.zeros(1, requires_grad=True)
print(b)
#y=0x+0

#가설 세우기
hypothesis = x_train * W + b
print(hypothesis)

#cost function 선언하기
cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)

#경사하강법, lr은 learning rate
optimizer = optim.SGD([W, b], lr=0.01)

#gradient를 0으로 초기화
optimizer.zero_grad()
#cost function을 미분하여 gradient 계산
cost.backward()
#W와 b를 업데이트
optimizer.step()

#전체 코드
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1): # epoch : 전체 훈련 데이터가 학습에 한 번 사용된 주기

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad() # 파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시킨다.
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

# 결과 : gold, label, gound-truth
'''
hyper-parameter : parameter 를 학습시키기 위한 parameter (ex. learning rate) 모델을 학습하기 전에 미리 지정해주는 값
hyper-parameter sweep : hyper-parameter를 조정하여 모델의 성능을 향상시키는 과정
    
    
    



'''