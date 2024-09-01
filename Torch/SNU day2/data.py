import os
import requests

if not os.path.exists("competition.py"):
    url = "https://link.teddynote.com/COMPT"
    file_name = "competition.py"
    response = requests.get(url)
    with open(file_name, "wb") as file:
        file.write(response.content)

import competition

# 파일 다운로드
# competition.download_competition_files(project)
competition.download_competition_files(
    "https://link.teddynote.com/ADVHOUSE", use_competition_url=False
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Data 경로 설정
DATA_DIR = "data"

# 경고 무시
warnings.filterwarnings("ignore")

SEED = 123

#데이터 로드
# train 데이터셋 로드 (train.csv)
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

# test 데이터셋 로드 (test.csv)
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

#기본 전처리
all_data = pd.concat([train, test], ignore_index=True)
all_data.head()

numeric_columns = all_data.select_dtypes(exclude="object")
numeric_columns

numeric_columns = numeric_columns.fillna(0)
numeric_columns

train_data = numeric_columns[: len(train)]
test_data = numeric_columns[len(train):]

#데이터셋 분할
x_train = train_data.drop("SalePrice", axis=1)
y_train = train_data["SalePrice"]
x_test = test_data.drop("SalePrice", axis=1)

x_train.head()

NUM_FEATURES = len(x_train.columns)

