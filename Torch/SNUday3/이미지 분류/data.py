import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

#내장 데이터셋 로드
# Image Transform 정의
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# train(학습용) 데이터셋 로드
train = datasets.FashionMNIST(
    root="data",
    train=True,  # set True
    download=True,  # 다운로드
    transform=transform,  # transform 적용. (0~1 로 정규화)
)

# test(학습용) 데이터셋 로드
test = datasets.FashionMNIST(
    root="data",
    train=False,  # set to False
    download=True,  # 다운로드
    transform=transform,  # transform 적용. (0~1 로 정규화)
)

#FashionMNIST 데이터셋 시각화
import matplotlib.pyplot as plt

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(12, 8))
cols, rows = 8, 5

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train), size=(1,)).item()
    img, label = train[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(torch.permute(img, (1, 2, 0)), cmap="gray")
plt.show()
