from data import *
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]  # 'FilePath' 열에서 이미지 경로 가져오기
        image = Image.open(img_path).convert("L")  # 이미지를 흑백으로 로드

        if self.transform:
            image = self.transform(image)

        return image


# 이미지를 불러올 때 적용할 전처리 정의
test_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),  # 이미지 크기를 (28, 28)로 조정
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    ]
)

# 커스텀 데이터셋 인스턴스 생성
custom_dataset = CustomImageDataset(dataframe=test, transform=test_transform)

# DataLoader 인스턴스 생성
test_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)