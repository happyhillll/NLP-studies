from datasets import load_dataset
from torch.utils.data import DataLoader  #SGD의 반복 연산을 실행할때 사용하는 미니 배치용 유틸리티 함수
import numpy as np
import torch

def get_testdata_loader():
    test_data = load_dataset('mnist', split='test') #mnist에서 split만 가져오겠다
    print()
    def transform_func(examples):
        examples['image']=[torch.FloatTensor(np.array(img)) for img in examples['image']] #examples의 image열에서 img를 vector로 변환
        return examples
    test_data=test_data.with_transform(transform_func)
    test_loader=DataLoader(test_data,batch_size=32, shuffle=True) #image를 벡터로 바꿔준 데이터가 들어가 있는 dataset을 batch size 32로 가져온다.
    return test_loader

if __name__ =="__main__":
    get_testdata_loader()


'''
테스트 방법
1. x_test를 torch float tensor로 변환
2. y_test를 mnist['label']로 변환
3. prediction=model(x_test)로 예측
4. prediction과 y_test를 비교
'''

def test():
    test_loader=get_testdata_loader()
    model=Classifier(28*28, 10)
    
    x_test=test_loader['image']
    y_test=test_loader['label']
    image=[torch.FloatTensor(np.array(img)) for img in x_test]
    for i in range(len(image)):
        image[i]=image[i].view(1,-1) 
    #y_test를 pytorch tensor로 변환
    y_test=torch.FloatTensor(np.array(y_test))
    
    #x_test 모델에 돌리고 y_test와 비교
    prediction=model(x_test)
    print(prediction)
    correct_prediction = torch.argmax(prediction, 1) == y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

if __name__ == "__main__":
    test()