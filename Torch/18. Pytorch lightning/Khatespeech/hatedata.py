'''
https://github.com/kocohub/korean-hate-speech/blob/master/guideline/annotation_guideline_en.md
'''

import koco

train_dev = koco.load_dataset('korean-hate-speech', mode='train_dev')
'''
train_dev['train'][0]
comments, contain_gender_bias(T/F), bias(gender, others, none), hate(hate, offensive, none), news_title
'''

unlabeled =koco.load_dataset('korean-hate-speech', mode='unlabeled')
test= koco.load_dataset('korean-hate-speech', mode='test')

# train 가져오기
class ds:
    def __init__(self):
        self.train_dev = train_dev['train']
        self.test_data = test
        
        self.x_train = []
        for i in range(len(self.train_dev)):
            self.x_train.append(self.train_dev[i]['comments'])
        self.y_train = []
        for i in range(len(self.train_dev)):
            self.y_train.append(self.train_dev[i]['hate'])
            
        self.x_test = []
        for i in range(len(self.test_data)):
            self.x_test.append(self.test_data[i]['comments'])

    def get_train(self):
        return self.x_train, self.y_train
    
    def get_test(self):
        return self.x_test
