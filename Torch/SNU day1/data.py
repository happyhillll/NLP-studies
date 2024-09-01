from opendata import dataset

dataset.download('toydata')

import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 보스톤 주택 가격 데이터셋 로드
df = pd.read_csv("data/toydata/boston_house_price.csv")
df.head()
print()

# feature(x), label(y)로 분할
x = df.drop("PRICE", axis=1)
y = df["PRICE"]

# feature 변수의 개수 지정
NUM_FEATURES = len(x.columns)

# feature 변수의 개수 지정
NUM_FEATURES = len(df.drop("PRICE", axis=1).columns)
print(f"number of features: {NUM_FEATURES}")
print()