'''
1. 데이터 다운로드
2. dataloader를 이용해서 데이터를 조금씩 가져오면서 모델에 넣어줘야 한다.
3. 모델에 넣어줄때는 벡터로 바꿔줘야 한다.
'''

from datasets import load_dataset
from torch.utils.data import DataLoader  #SGD의 반복 연산을 실행할때 사용하는 미니 배치용 유틸리티 함수
import numpy as np
import torch

def get_data_loader():
    dataset = load_dataset('mnist', split='train') #mnist에서 split만 가져오겠다
    print()
    def transform_func(examples):
        examples['image']=[torch.FloatTensor(np.array(img)) for img in examples['image']] #examples의 image열에서 img를 vector로 변환
        return examples
    dataset=dataset.with_transform(transform_func)
    loader=DataLoader(dataset,batch_size=32, shuffle=True) #image를 벡터로 바꿔준 데이터가 들어가 있는 dataset을 batch size 32로 가져온다.
    return loader

if __name__=="__main__":
    get_data_loader()
