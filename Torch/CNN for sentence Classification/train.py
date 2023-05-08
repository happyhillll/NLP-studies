from ds import SenCLSDataset
from torch.utils.data import DataLoader
from model import SenCNN
from vocab import get_vocab

data = [
    ("나는 나는 밥이 좋아", 0), ("나는 나는 니가 실허", 1), ("나는 니가 니가 싫어", 1)
]

vocab = get_vocab()
dataset = SenCLSDataset(data, vocab)
loader = DataLoader(dataset, batch_size=2)
model = SenCNN()

for i, batch in enumerate(loader):
    x, y = batch
    hat_y = model(x)  # [bsz, vocab_len]
    # TODO y, hat_y 이 창리ㅡㄹ loss를 잘 구해서 학습시키기