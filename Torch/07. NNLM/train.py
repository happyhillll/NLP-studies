# train.py
from ds import BiGramDataset
from torch.utils.data import DataLoader
from model import BiGramLM
from vocab import get_vocab

train_doc = '''
    [서울=뉴시스] 양소리 기자 = 윤석열 대통령은 28일(현지시간) 미국 보스턴의 하버드대학교에서 연설을 끝으로 공식 일정을 마무리했다.
    지난 27일 저녁 보스턴에 도착한 윤 대통령은 28일 하루동안 매사추세츠공과대학(MIT) 석학과의 간담회, 한미 클러스터 라운드 테이블, 메사추세츠 주지사와의 오찬, 하버드대 연설 등 일정을 소화했다.
    윤 대통령은 29일 오전 귀국길에 오른다.
    '''

vocab = get_vocab()
dataset = BiGramDataset(train_doc, vocab) 
loader = DataLoader(dataset, batch_size=4)
model = BiGramLM(len(vocab), 5)

for i, batch in enumerate(loader):
    x, y = batch
    hat_y = model(x)  # [bsz, vocab_len]
    # TODO y, hat_y 차이의 loss를 잘 구해서 학습시키기