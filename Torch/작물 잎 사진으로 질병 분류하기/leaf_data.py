'''
data : https://data.mendeley.com/datasets/tywbtsjrjv/1
목표 : 작물 잎 사진을 통해 해당 작물이 질병을 가지고 있는지의 여부를 판단. 
데이터 : 작물의 이름_질병종류 or healthy

1. 데이터 다운로드
2. 데이터를 train, test로 분할
3. dataloader를 이용해서 데이터를 조금씩 가져오면서 모델에 넣어주기
4. 모델에 넣어줄때는 벡터로 바꿔줘야 한다.
'''

from torch.utils.data import DataLoader
import numpy as np
import torch

#원본데이터셋 어디있는지 찾기
import os
import shutil

#data loader에 넣어서 돌리기
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def get_data_loader():
    transform_base = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]) 
    train_dataset = ImageFolder(root='/Users/minjikim/GitHub/NLP-studies/Torch/작물 잎 사진으로 질병 분류하기/splitted/train', transform=transform_base)
    test_dataset = ImageFolder(root='/Users/minjikim/GitHub/NLP-studies/Torch/작물 잎 사진으로 질병 분류하기/splitted/test', transform=transform_base)
    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)
    return train_loader, test_loader

if __name__=="__main__":
    
    original_dataset_dir='/Users/minjikim/Downloads/dataset'
    classes_list = os.listdir(original_dataset_dir) #os.listdir(): 해당 경로 하위에 있는 모든 폴더의 목록을 가져오는 메서드

    #나눈 데이터를 저장할 폴더 생성
    base_dir='/Users/minjikim/GitHub/NLP-studies/Torch/작물 잎 사진으로 질병 분류하기/splitted' 
    os.mkdir(base_dir)

    #분리 후에 각 데이터를 저장할 하위폴더 train, test를 생성
    train_dir=os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    test_dir=os.path.join(base_dir,'test')
    os.mkdir(test_dir)

    #train, test 폴더 하위에 각각 클래스 목록 폴더를 생성
    for clss in classes_list:
        os.mkdir(os.path.join(train_dir,clss))
        os.mkdir(os.path.join(test_dir, clss))

    #train, test 데import math

    import math
    
    for cls in classes_list:
        path = os.path.join(original_dataset_dir, cls)
        fnames = os.listdir(path)
    
        train_size = math.floor(len(fnames) * 0.8)
        test_size = math.floor(len(fnames) * 0.2)
        
        train_fnames = fnames[:train_size]
        print("Train size(",cls,"): ", len(train_fnames))
        for fname in train_fnames:
            src = os.path.join(path, fname)
            dst = os.path.join(os.path.join(train_dir, cls), fname)
            shutil.copyfile(src, dst)
            
            
        test_fnames = fnames[train_size:(train_size +test_size)]

        print("Test size(",cls,"): ", len(test_fnames))
        for fname in test_fnames:
            src = os.path.join(path, fname)
            dst = os.path.join(os.path.join(test_dir, cls), fname)
            shutil.copyfile(src, dst)

        
    
    get_data_loader()