
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#랜덤 시드 고정
torch.manual_seed(1)

#훈련 데이터 선언
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

#가중치 W와 편향 b의 초기화 : requires_grad=True로 설정하여 학습을 통해 값이 변경되도록 함
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b= torch.zeros(1, requires_grad=True)

#가설, loss 함수, 옵티마이저를 선언한 후에 경사 하강법 1000회 수행
#optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    
    #hypothesis 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    
    #loss 계산
    loss = torch.mean((hypothesis - y_train) ** 2)
    
    #loss로 H(x) 개선
    optimizer.zero_grad() #gradient를 0으로 초기화
    loss.backward()  #loss를 미분하여 gradient 계산
    optimizer.step() #gradient를 사용하여 가중치 업데이트
    
    #100번마다 로그 출력
    if epoch % 100 == 0: #epoch가 100으로 나누어 떨어질 때
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Loss: {:.6f}'.format( 
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), loss.item()
        )) 
print(b.grad)
print(w1.grad)
print(w2.grad)
print(w3.grad)


#행렬 연산을 이용한 방법
#이번에는 x_train 하나에 모든 샘플을 담음
x_train = torch.FloatTensor([[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

print(x_train.shape)
print(y_train.shape)

#가중치와 편향의 초기화
W= torch.zeros((3, 1), requires_grad=True)
b= torch.zeros(1, requires_grad=True)
# x_train과 W를 곱하려면 W는 (3,1)이어야 함

hypothesis = x_train.matmul(W) + b

# 전체 코드 
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward() #왜 이름이 backward인가?
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))

print(b.grad)
print(W.grad)