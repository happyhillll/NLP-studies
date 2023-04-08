'''
1. 벡터, 행렬, 텐서
2. 넘파이 훑어보기
3. 파이토치 텐서 선언하기
4. 행렬 곱셈
5. 다른 오퍼레이션들
'''

# 2. 넘파이 훑어보기
import numpy as np

# 1) 1D with numpy
t=np.array([0.,1.,2.,3.,4.,5.,6.])
print(t)

# 차원과 벡터 크기 출력
print('Rank of t:',t.ndim)
print('Shape of t:',t.shape) # (5,)의 형식은 (1,5)와 같다.

# 1-1) Numpy 기초 이해하기
print('t[0] t[1] t[-1] =',t[0],t[1],t[-1])
print('t[2:5] t[4:-1] =',t[2:5],t[4:-1])
print('t[:2] t[3:] =',t[:2],t[3:])

# 2) 2D with numpy
t=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
print(t)

print('Rank of t:',t.ndim)
print('Shape of t:',t.shape) #행렬이 4x3이라는 것을 알 수 있다.

# 파이토치 텐서 선언하기 : 파이토치는 넘파이와 매우 유사하지만 더 낫다.
import torch

 # 1) 1D with Pytorch
 t=torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
 print(t)
 
 print(t.dim()) # rank : 차원
 print(t.shape) # shape : 크기
 print(t.size()) # shape : 크기
 
 print(t[0],t[1],t[-1])
 print(t[2:5],t[4:-1])
 print(t[:2],t[3:])
 
 #2) 2D with Pytorch
 t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)
 
 print(t.dim()) # rank : 차원
 print(t.shape) # shape : 크기
 print(t.size()) # shape : 크기
 
 print(t[:,1]) # 첫번째 차원을 전체 선택하고 두번째 차원에서 1번째 인덱스만 선택
 print(t[:,1].size()) # 크기는 1차원 벡터
 
 print(t[:,:-1]) # 첫번째 차원을 전체 선택하고 두번째 차원에서 마지막 인덱스를 제외한 모든 인덱스 선택
 
 # 3) 브로드캐스팅
 m1 = torch.FloatTensor([[3, 3]])
 m2 = torch.FloatTensor([[2, 2]])
 print(m1+m2)
 
 # Vector + Scalar
 m1=torch.FloatTensor([[1,2]])
 m2=torch.FloatTensor([3]) # 3 -> [[3,3]]
 print(m1+m2)
 
 #2x1 vector + 1x2 vector : 파이토치에서는 두 벡터의 크기를 (2,2)로 변경하여 더할 수 있다.
 m1=torch.FloatTensor([[1,2]])
 m2=torch.FloatTensor([[3],[4]])
 print(m1+m2)
 
# 4) 자주 사용되는 기능들
# 1_ 행렬 곱셈과 곱셈의 차이
m1=torch.FloatTensor([[1,2],[3,4]])
m2=torch.FloatTensor([[1],[2]])
print('Shape of Matrix 1:',m1.shape)
print('Shape of Matrix 2:',m2.shape)
print(m1.matmul(m2)) # 행렬 곱셈


#element-wise 곱셈에서는 브로드캐스팅이 된 후에 곱셈이 수행된다.
m1=torch.FloatTensor([[1,2],[3,4]])
m2=torch.FloatTensor([[1],[2]])
print('Shape of Matrix 1:',m1.shape)
print('Shape of Matrix 2:',m2.shape)
print(m1*m2) # element-wise 곱셈 : 동일한 크기의 행렬이 동일한 위치에 있는 원소끼리 곱하는 것
print(m1.mul(m2))

# 2_ 평균
t=torch.FloatTensor([1,2])
print(t.mean())

t=torch.FloatTensor([[1,2],[3,4]])
print(t.mean()) # 4개 원소의 평균이 나옴

print(t.mean(dim=0)) # 첫번째 차원(행)을 제거하고 평균을 구함 
print(t.mean(dim=1))
print(t.mean(dim=-1)) # 마지막 차원을 제거하고 평균을 구함 = 열의 차원 제거

# 3_ 덧셈
print(t.sum()) # 모든 원소의 합
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

# 4_ 최대(Max)와 아그맥스(Argmax): 최대값을 가진 인덱스
print(t.max()) # 최대값
print(t.max(dim=0)) # 첫번째 차원(행)을 제거하고 최대값을 구함

print('Max:  ',t.max(dim=0)[0]) #max값만 받아오기
print('Argmax:',t.max(dim=0)[1]) #argmax값만 받아오기

print(t.max(dim=1)) 
print(t.max(dim=-1)) # 마지막 차원을 제거하고 최대값을 구함 = 열의 차원 제거

import torch
