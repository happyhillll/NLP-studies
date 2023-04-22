'''

view : tensor의 shape을 변경


- getloader : 데이터를 불러오는 함수 먼저 만들기
- nn.module 만들기
- cross entropy loss 쓰기

- batchsize라고 안하고 example 개수라고 하는게 맞음
- view로 [n_sample, -1] 로 바꿔주기

'''


#데이터 읽기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #최적화 함수
import set_matplotlib_korean
from torchvision import datasets, transforms 
from matplotlib import pyplot as plt

#cuda가 가능하다면 cuda를 사용하고, 아니라면 cpu를 사용
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('Current cuda device is', device)

#파라미터 설정
batch_size = 50
learning_rate = 0.0001
epoch_num = 15

train_data = datasets.MNIST(root = './data/02/',
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())
print() #디버깅

test_data = datasets.MNIST(root = './data/02/',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())
print('number of training data : ', len(train_data))
print('number of test data : ', len(test_data))

#데이터 확인
image, label = train_data[0]
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title('label : %s' % label)
plt.show()

#시작
#텐서데이터셋
from torch.utils.data import TensorDataset
#데이터로더
from torch.utils.data import DataLoader

dataset=TensorDataset(image, label)
#아 이미 이렇게 구성되어져 있는건가???

#데이터 로더
dataloader=DataLoader(train_data,batch_size=2,shuffle=True)

print(image.shape) #[1,28,28] : 1차원으로 만들어야 함.. > 앞이 1인 이유는 28*28의 example이 1개라는 뜻

#모델과 옵티마이저 설계
model = nn.Linear(3,1) #data의 shape을 알아야 하는디
