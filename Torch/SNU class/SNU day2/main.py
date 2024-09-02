from torch.utils.data import DataLoader
from ds import *
from data import *
from model import Net
import matplotlib.pyplot as plt
import os

train_dataset=CustomDataset(x_train,y_train, normalize=True)
test_dataset=CustomDataset(x_test, normalize=True)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=1, shuffle=False)


model=Net(NUM_FEATURES)
losses=model.train_model(train_loader,num_epoch=100)

# 결과입력
your_answer = []

# evaluation mode 로 변경
model.eval()

device="mps"

with torch.no_grad():
    for x in test_loader:
        x = x.to(device)
        y_hat = model(x).cpu().detach().numpy().item()
        your_answer.append(y_hat)

import competition

# 예측 결과 업데이트
from IPython.display import display

project="ADVHOUSE"
username= "happyhill@snu.ac.kr"
password= "1234"

submission = pd.read_csv(os.path.join(DATA_DIR, "submission.csv"))
submission["SalePrice"] = your_answer

display(submission)
competition.submit(project, username, password, submission)