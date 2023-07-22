import json

class ds:
    def __init__(self):
        self.train_data = []

        with open("//Users/minjikim/GitHub/NLP-studies/Torch/Hate Speech Detection/NIKL_AU_2023_COMPETITION_v1.0/nikluge-au-2022-train.jsonl","r",encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.train_data.append(data)

        self.x_train=list(utterance['input'] for utterance in self.train_data)
        self.y_train=list(label['output'] for label in self.train_data)
        
        self.dev_data = []

        with open("/Users/minjikim/GitHub/NLP-studies/Torch/Hate Speech Detection/NIKL_AU_2023_COMPETITION_v1.0/nikluge-au-2022-dev.jsonl","r",encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.dev_data.append(data)

        self.x_dev=list(utterance['input'] for utterance in self.dev_data)
        self.y_dev=list(label['output'] for label in self.dev_data)
    
    def get_train(self):
        return self.x_train, self.y_train
    
    def get_dev(self):
        return self.x_dev, self.y_dev
         
# ds=ds()
# print(ds)