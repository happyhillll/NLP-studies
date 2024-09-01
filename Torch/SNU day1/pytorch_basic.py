import torch
device = torch.device("mps")

import torch
import numpy as np

print(torch.__version__)

arr = np.array([1,3,5,7,9])
print(arr)
print(arr.dtype)
print(type(arr))

#sharing
t1=torch.from_numpy(arr)
print(t1)
print(t1.dtype)
print(t1.type())
print(type(t1))

t2 = torch.as_tensor(arr)
print(t2)  # 출력
print(t2.dtype)  # dtype은 데이터 타입
print(t2.type())  # type()은 텐서의 타입
print(type(t2))  # t2 변수 자체의 타입

# numpy array의 0번 index를 999로 값 변환
arr[0] = 999

# t1, t2 출력
print(f"t1: {t1}")
print(f"t2: {t2}")

#copying
# 샘플 데이터 초기화
arr = np.array([1, 3, 5, 7, 9])
print(arr)
print(arr.dtype)
print(type(arr))

t3 = torch.tensor(arr)
print(t3)  # 출력
print(t3.dtype)  # dtype은 데이터 타입
print(t3.type())  # type()은 텐서의 타입
print(type(t3))  # t3 변수 자체의 타입

# numpy array의 0번 index를 999로 값 변환
arr[0] = 999
