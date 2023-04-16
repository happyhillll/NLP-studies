# requires_grad=True : 자동 미분 기능 적용, backward()를 통해 gradient 계산 = 자동 미분 기능을 수행하고 있는 것임

import torch

# 값이 2인 임의의 스칼라 텐서 w 선언 (requires_grad=True로 설정하여 연산 기록을 추적)
w=torch.tensor(2.0, requires_grad=True)

#수식 정의
y=w**2
z=2*y+5

#해당 수식의 w에 대한 기울기 계산
z.backward()

print('수식을 w로 미분한 값 : {}'.format(w.grad))

