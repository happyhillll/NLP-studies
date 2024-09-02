import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

project = "SIMPSON"  # 수정하지 마세요
username = "happyhill@snu.ac.kr"  # 이메일아이디 (예시. abc@hello.com)
password = "1234"  # 비밀번호

# Data 경로 설정
DATA_DIR = "Torch/SNU class/SNU day3/이미지 분류"

# 경고 무시
warnings.filterwarnings("ignore")

SEED = 123

#데이터 로드
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test["FilePath"] = test["FilePath"].str.replace("./test_data/", "Torch/SNU class/SNU day3/이미지 분류/test/")
test.head()

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 이미지 전처리 정의
train_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),  # 이미지 크기를 (28, 28)로 조정
        transforms.Grayscale(num_output_channels=1),  # 흑백 이미지로 변환
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    ]
)

# ImageFolder를 사용하여 데이터셋 생성
train_dataset = datasets.ImageFolder(root="Torch/SNU class/SNU day3/이미지 분류/train", transform=train_transform)

# DataLoader 인스턴스 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 숫자 라벨을 클래스 이름으로 매핑하는 딕셔너리 생성
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# train_loader에서 첫 번째 배치 가져오기
images, labels = next(iter(train_loader))

# 4x8 그리드로 이미지와 라벨 시각화
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(16, 8))

# 각 subplot에 이미지와 라벨을 그리기
for ax, img, label in zip(axes.flat, images, labels):
    ax.imshow(img.numpy().squeeze(), cmap="gray")  # 이미지를 그레이스케일로 표시
    ax.set_title(
        idx_to_class[label.item()], fontsize=12
    )  # 각 이미지 위에 라벨(클래스 이름)을 타이틀로 추가
    ax.axis("off")  # 축 제거

# plt.tight_layout()  # subplot 간격 조정
# plt.show()