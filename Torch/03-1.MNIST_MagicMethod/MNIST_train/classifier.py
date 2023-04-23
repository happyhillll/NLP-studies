'''
1. 모델 설계
'''
import torch.nn as nn #신경망 모듈. 매개변수를 캡슐화(encapsulation)하는 간편한 방법 으로, GPU로 이동, 내보내기(exporting), 불러오기(loading) 등의 작업을 위한 헬퍼(helper)를 제공합니다.

#신경망 모듈. 매개변수를 캡슐화(encapsulation)하는 간편한 방법 으로, GPU로 이동, 내보내기(exporting), 불러오기(loading) 등의 작업을 위한 헬퍼(helper)를 제공합니다.
class Classifier(nn.Module): #nn.Module을 상속받아왔음
    def __init__(self, in_dim, out_dim): # init도 같이 상속 받으려면 정의해서 super를 넣어주어야 함.
        super().__init__() 
        self.linear=nn.Linear(in_dim,out_dim) #in_dim : 입력 차원, #out_dim : 출력 차원
        
    def forward(self,x):  
        return self.linear(x)