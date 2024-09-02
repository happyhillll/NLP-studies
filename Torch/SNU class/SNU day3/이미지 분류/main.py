from data import *
from model import *
from ds import *
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from IPython.display import display
import os
import requests
import competition

if not os.path.exists("competition.py"):
    url = "https://link.teddynote.com/COMPT"
    file_name = "competition.py"
    response = requests.get(url)
    with open(file_name, "wb") as file:
        file.write(response.content)

device="mps"

#dataset
custom_dataset=CustomImageDataset(dataframe=test, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

# 모델 생성
model = Net(num_classes=10)
losses=model.train_model(train_loader)

#예측
predictions = []

# 검증모드 진입
model.eval()

with torch.no_grad():
    # loss 초기화
    running_loss = 0
    # 정확도 계산
    running_acc = 0
    for x in test_loader:
        x = x.to(device)

        y_hat = model(x)
        label = y_hat.argmax(dim=1).detach().item()
        predictions.append(label)
        
# 정답
your_answer = [idx_to_class[l] for l in predictions]

#결과 예측

# 예측 결과 업데이트
submission = pd.read_csv(os.path.join(DATA_DIR, "submission.csv"))
submission["Label"] = your_answer

display(submission)
competition.submit(project, username, password, submission)