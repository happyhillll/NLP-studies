from data import *
import torch
import numpy as np


batch_size = 32  # batch_size 지정
num_workers = 8  # Thread 숫자 지정 (병렬 처리에 활용할 쓰레드 숫자 지정)

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=batch_size, shuffle=False, num_workers=num_workers
)