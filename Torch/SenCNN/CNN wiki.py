'''
CNN
- 이미지 처리에 탁월한 성능
- 목표 : CNN으로 텍스트 처리하기
- 합성곱층 : Conv(합성곱 연산)+ReLU(활성화 함수)
- 풀링층 : 풀링 연산

- CNN 사용하는 이유
    - overfitting
    - 이미지를 다층 퍼셉트론으로 분류하면 픽셀을 1차원인 벡터로 변환함. 
    - but 공간적인 구조 정보가 유실된 상태임
    - 공간적인 구조 정보 : 거리가 가까운 픽셀들끼리는 어떤 연관이 있고,
        어떤 픽셀들끼리는 값이 비슷하고.. 보존하면서 학습해야 정확!

- 채널 : 깊이 
- 합성곱층 : 합성곱 연산으로 이미지의 특징 추출
'''
def get_vocab():
    train_data = 'wait for the video and don''t rent it'
    

list(set(train_data.split()))