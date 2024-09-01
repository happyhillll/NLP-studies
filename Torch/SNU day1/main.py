from torch.utils.data import DataLoader
from ds import *
from data import *
from model import Net
import matplotlib.pyplot as plt

 # Custom으로 정의한 데이터셋 생성
dataset = CustomDataset(df, "PRICE", True)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
x,y=next(iter(data_loader))

model=Net(NUM_FEATURES)
losses=model.train_model(data_loader,num_epoch=200)

# 전체 loss 에 대한 변화량 시각화
plt.figure(figsize=(14, 6))
plt.plot(losses[:100], c="darkviolet", linestyle=":")

plt.title("Losses over epoches", fontsize=15)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
print()