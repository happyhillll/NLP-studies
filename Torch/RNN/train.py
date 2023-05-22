from ds import SenCLSDataset
from torch.utils.data import DataLoader
from SenRNN import SenRNN
from word2vec import get_w2v
from data import train_data, test_data


dataset = SenCLSDataset(train_data, w2v)
loader = DataLoader(dataset, batch_size=2)
model = SenRNN()

for i, batch in enumerate(loader):
    x, y = batch
    hat_y = model(x)  # [bsz, vocab_len]

