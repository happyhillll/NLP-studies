from ds import nsmc_SentDataset
from torch.utils.data import DataLoader
from RNN import Model
from nsmc_data import nsmc_data
import urllib.request
import pandas as pd

#
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
train_data = pd.read_table('ratings_train.txt')
train_data = train_data.dropna(subset=['document'],axis=0)
data = train_data[['document', 'label']].apply(tuple, axis=1).tolist()

vocab = nsmc_data()
dataset = nsmc_SentDataset(data, vocab)
loader = DataLoader(dataset, batch_size=2)
model = Model()

for i, batch in enumerate(loader):
    x, y = batch
    hat_y = model(x)  # [bsz, vocab_len]
    # TODO y, hat_y 이 창리ㅡㄹ loss를 잘 구해서 학습시키기