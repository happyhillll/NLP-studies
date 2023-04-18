'''
1. 미니 배치와 배치 크기
- 미니 배치: 전체 데이터 중 일부만 사용하여 학습하는 것
- 미니 배치 학습을 하게 되면 미니 배치 만큼만 가져가서 미니 배치에 대한 손실을 계산하고, 경사 하강법을 수행한다.
- 에포크 : 전체 데이터를 한 번 학습하는 것
- 배치 경사 하강법 : 전체 데이터를 사용하여 경사 하강법을 수행하는 것
- 미니 배치 경사 하강법 : 전체 데이터 중 일부만 사용하여 경사 하강법을 수행하는 것
- 배치 크기 : 보통 2의 제곱수를 사용한다. (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
    - 왜냐하면 CPU와 GPU의 메모리가 2의 배수이므로 배치크기가 2의 제곱수일 경우에 데이터 송수신의 효율을 높일 수 있다.

2. 이터레이션
- 한번의 에포크 내에서 이루어지는 매개변수인 가중치 W와 편향 b의 업데이트 횟수


'''

#3. 데이터 로드하기
import torch
import torch.nn as nn
import torch.nn.functional as F

#텐서데이터셋
from torch.utils.data import TensorDataset
#데이터로더
from torch.utils.data import DataLoader

x_train = torch.FloatTensor([[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

dataset = TensorDataset(x_train, y_train) # 텐서데이터셋의 입력으로 텐서를 묶어준다.
dataloader=DataLoader(dataset, batch_size=2, shuffle=True) # 데이터로더의 입력으로 텐서데이터셋을 넣어준다.
# 데이터 로더는 기본적으로 2개의 인자를 입력받음
# 첫번째 인자: 데이터셋, 두번째 인자: 배치 크기 > 미니 배치의 크기는 통상적으로 2의 배수를 사용한다., shuffle=True > 데이터를 섞어서 불러온다.

# 모델과 옵티마이저 설계
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    print(batch_idx)
    print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        )) 
