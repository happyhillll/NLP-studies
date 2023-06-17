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

print(t.max(dim=1)
print(t.max(dim=-1)) # 마지막 차원을 제거하고 최대값을 구함 = 열의 차원 제거

# 4) 뷰(View) : 텐서의 크기(size)나 차원(dim)을 변경할 때 사용
# 우선 임의로 3차원 텐서 만들기
t = np.array([[[0,1,2],
              [3,4,5]],
              [[6,7,8],
              [9,10,11]]])
ft=torch.FloatTensor(t)
print(ft.shape) 

#4-1) 3차원 텐서에서 2차원 텐서로 변경
print(ft.view([-1,3])) # -1은 다른 차원들을 이용해서 유추한다는 의미 : ft라는 텐서를 (?,3)의 크기로 변경
print(ft.view([-1,3]).shape)

'''
- view는 텐서의 크기를 변경하는 연산이지만, 텐서 안의 원소의 개수는 유지되어야 한다.
- view는 사이즈가 -1로 설정되면 다른 차원을 이용해서 유추한다.
'''

# 4-2) 3차원 텐서는 유지하되, 크기를 바꾸기.
print(ft.view([-1,1,3]))
print(ft.view([-1,1,3]).shape)

# 5) 스퀴즈(Squeeze) : 텐서에서 1인 차원을 제거
# 임의로 2차원 텐서 만들기
ft=torch.FloatTensor([[0],[1],[2]])
print(ft.shape)

print(ft.squeeze()) # 1인 차원을 제거
print(ft.squeeze().shape)

#6) 언스퀴즈(Unsqueeze) : 텐서에 1인 차원을 추가
ft=torch.FloatTensor([0,1,2])
ft=ft.unsqueeze(0) # 0번째 차원에 1인 차원을 추가
print(ft.shape)

print(ft.unsqueeze(1).shape) # 1번째 차원에 1인 차원을 추가
print(ft.unsqueeze(-1).shape) # 마지막 차원에 1인 차원을 추가

#7) 타입 캐스팅(Type Casting)
lt=torch.LongTensor([1,2,3,4])
print(lt)

print(lt.float()) # long타입을 float타입으로 변경

bt=torch.ByteTensor([True,False,False,True])
print(bt)

print(bt.long()) # byte타입을 long타입으로 변경
print(bt.float()) # byte타입을 float타입으로 변경

#8) 연결하기(concatenate)
x=torch.FloatTensor([[1,2],[3,4]])
y=torch.FloatTensor([[5,6],[7,8]])

print(torch.cat([x,y],dim=0)) # dim=0 : 행을 기준으로 연결
print(torch.cat([x,y],dim=1)) # dim=1 : 열을 기준으로 연결

#9) 스택킹(Stacking) : 연결을 하는 또 다른 방법
x=torch.FloatTensor([1,4])
y=torch.FloatTensor([2,5])
z=torch.FloatTensor([3,6]) #셋 다 scalar값이므로 1차원 텐서

print(torch.stack([x,y,z])) # dim=0 : 행을 기준으로 연결
#사실 많은 연산을 한 번에 축약하고 있음
print(torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)],dim=0))
print(torch.stack([x,y,z],dim=1)) # dim=1 : 열을 기준으로 연결

#10) ones_like & zeros_like : 0으로 채워진 텐서와 1로 채워진 텐서
x=torch.FloatTensor([[0,1,2],[2,1,0]])
print(x)

print(torch.ones_like(x)) # x와 같은 크기의 1로 채워진 텐서
print(torch.zeros_like(x)) # x와 같은 크기의 0으로 채워진 텐서

#11) In-place Operation(덮어쓰기 연산)
x=torch.FloatTensor([[1,2],[3,4]])
print(x.mul(2.)) # 2를 곱한 새로운 텐서를 반환
print(x) # x는 변하지 않음

print(x.mul_(2.)) # 언더바를 쓰면 덮어쓸 수 있음
print