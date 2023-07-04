mention : 특정 개체를 나타내는 표현
span : 연속된 토큰으로 이루어지는 토큰의 집
num of span : T(T+1)/2

상호참조해결 : 문서 내에서 발생한 mention들 중 같은 개체를 의미하는 mentio들ㅇ,ㄹ 묶는 문제
1. 뭐가 멘션인가? : 위치, span만 찾는거임
2. 멘션끼리 clustering : 같은거 끼리

- 모든 멘션이 될 수 있는 spans을 candidates로 두고, 
scoring을 통해 상위 k개를 mention으로 간주. k: 문서 길이에 비례

1. 뭐가 멘션인가? 
- attention은 앞에랑 맨 마지막이 더 클 확률이 있음

2. 멘션끼리 clustering 
-  