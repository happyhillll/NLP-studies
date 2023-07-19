import json

class ds:
    def __init__(self):
        self.x_train=['너는 정말 예쁜거 가탕','너 진짜 못생겼구나','진짜 귀엽고 멋지네']
        self.y_train=[1,0,1]
        
        self.x_dev=['아유 이뻐','아유 못생겼어','못난놈이네']
        self.y_dev=[1,0,0]
    
    def get_train(self):
        return self.x_train, self.y_train
    
    def get_dev(self):
        return self.x_dev, self.y_dev
         
# ds=ds()
# print(ds)