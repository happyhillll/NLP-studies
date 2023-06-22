# ds.py

from torch.utils.data import Dataset
import torch
import random


class DialogDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.x, self.y = [], []
        self.preprocess(data, tokenizer)

    def preprocess(self, datas, tokenizer):

        labels = list(set([d['label'] for d in datas]))
        label_mapper = {topic: index for index, topic in enumerate(labels)}

        for data in datas:
            x = tokenizer(data["text"], padding="max_length")
            input_ids = x.input_ids
            attention_mask = x.attention_mask
            token_type_ids = x.token_type_ids

            cur_input_ids, cur_attention_mask, cur_token_type_ids = [], [], []
            st = 0
            while True:
                en = st + 512
                cur_input_ids.append(input_ids[st:en])
                cur_attention_mask.append(attention_mask[st:en])
                cur_token_type_ids.append(token_type_ids[st:en])
                st += 512
                if st >= len(x.input_ids): break

            cur_input_ids[-1] = cur_input_ids[-1] + [0]*(512-len(cur_input_ids[-1]))
            cur_attention_mask[-1] = cur_attention_mask[-1] + [0] * (512 - len(cur_attention_mask[-1]))
            cur_token_type_ids[-1] = cur_token_type_ids[-1] + [0] * (512 - len(cur_token_type_ids[-1]))

            cur_input_ids = cur_input_ids[:4]
            cur_attention_mask = cur_attention_mask[:4]
            cur_token_type_ids = cur_token_type_ids[:4]

            if len(cur_input_ids) < 4:
                for _ in range(4 - len(cur_input_ids)):
                    cur_input_ids.append([0]*512)
                    cur_attention_mask.append([0] * 512)
                    cur_token_type_ids.append([0] * 512)
            try:
                self.x.append(
                    {
                        "input_ids": torch.LongTensor(cur_input_ids),
                        "attention_mask": torch.LongTensor(cur_attention_mask),
                        "token_type_ids": torch.LongTensor(cur_token_type_ids),
                    }
                )
                self.y.append(label_mapper[data['label']])
            except:
                print()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



class NSMCDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.preprocess(data, tokenizer)

    def preprocess(self, datas, tokenizer):
        """
        :param datas: list of (id, text, label)
        :param tokenizer:
        :return:
        """
        self.x, self.y = [], []

        texts = [d[1] for d in datas]
        labels = [int(d[2]) for d in datas]
        """
        texts, labels = [], []
        for d in datas:
            texts.append(d[1])
            labels.append(int(d[2]))
        """

        tokens = tokenizer(texts, truncation=True, padding="max_length")


        tokens.input_ids = torch.LongTensor(tokens.input_ids)
        tokens.attention_mask = torch.LongTensor(tokens.attention_mask)
        tokens.token_type_ids = torch.LongTensor(tokens.token_type_ids)


        for i in range(len(tokens.input_ids)):
            token = {
                'input_ids': tokens.input_ids[i],
                'attention_mask': tokens.attention_mask[i],
                'token_type_ids': tokens.token_type_ids[i],
            }
            self.x.append(token)

        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]