# ds.py
from torch.utils.data import Dataset
import torch
import random
from bertNERdata import *

class NERDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.preprocess(data, tokenizer) #Dataset을 실행하면 자동으로 preprocess 실행
        # self.convert_to_features(texts, unique_values, tokenizer)
        
    def preprocess(self, datas, tokenizer):
        texts=ds().get_train()[0][:1000]
        label=ds().get_train()[1][:1000]
        # 데이터 쪼개기
        split_data = [d.split() for d in label]
        # 중복되지 않은 값 모으기
        unique_values = list(set([value for sublist in split_data for value in sublist]))
    
    # def convert_to_features (self, texts, unique_values, tokenizer, max_seq_len=-100, unk_token=-100,cls_token_segment_id=0,
    #                              pad_token_segment_id=0,
    #                              sequence_a_segment_id=0,
    #                              mask_padding_with_zero=True):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token #cls_token : [CLS]
        sep_token = tokenizer.sep_token #sep_token : [SEP]
        unk_token = tokenizer.unk_token #unk_token : [UNK]
        pad_token_id = tokenizer.pad_token_id #pad_token_id : 0
        
        features=[]
        vocab = {ner: idx for idx, ner in enumerate(unique_values)}
        unk_token = -100
        vocab[unk_token] = len(vocab)  # Add the [UNK] token to the last index of the vocab
        
        slot_labels = []
        for words in split_data:
            labels = []
            for word in words:
                label = vocab.get(word, vocab[unk_token])
                labels.append(label)
            slot_labels.append(labels)
        
        label = [', '.join(map(str, labels)) for labels in slot_labels]

        texts = [text.split() for text in texts]
        texts = [', '.join(words) for words in texts]
        label = [label for label in label]
        
         # Tokenize word by word (for NER)/ 여기 75개로 줄어드는데 왜그러는건지?⭐️
        tokens = []
        label_ids = []
        for word, slot_label in zip(texts, label):
            word_tokens = tokenizer.tokenize(word)  #texts에서 tokenize를 하는게 맞는지?
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            max_seq_len = max([len(word_tokens) for i in word_tokens])
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([int(label) for label in slot_label.split(',')] + [unk_token] * (len(word_tokens) - 1))
            # 단어 토큰의 첫번째 토큰에는 실제 레이블을 사용하고, 나머지는 패딩 레이블로 추가
            
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            label_ids = label_ids[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        sequence_a_segment_id = 0
        tokens += [sep_token]
        label_ids += [unk_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        cls_token_segment_id=0
        tokens = [cls_token] + tokens
        label_ids = [unk_token] + label_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
         # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        mask_padding_with_zero=True
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        pad_token_segment_id=0
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        pad_token_label_id=-100
        label_ids = label_ids + ([pad_token_label_id] * padding_length)
        
        # features에 input_ids, attention_mask, token_type_ids, label_ids를 넣어준다.
        
        return features
        
        # # vocab={
        # #     word : idx for idx, word in enumerate(labels)
        # # }
        # # labelss=[]
        # # for i in label:
        # #     if i in vocab.keys():
        # #         labelss.append(vocab[i])
        
        # # vocab={
        # #     word : idx for idx, word in enumerate(labels)
        # # }
        # # labelss=[]
        # # labelss.append(vocab[x] for x in label)
        
        # tokens.input_ids = torch.LongTensor(tokens.input_ids)
        # tokens.attention_mask = torch.LongTensor(tokens.attention_mask)
        # tokens.token_type_ids = torch.LongTensor(tokens.token_type_ids)
        
        # self.x, self.y = [], []
        
        # for i in range(len(tokens.input_ids)):
        #     token={
        #         'input_ids': tokens.input_ids[i],
        #         'attention_mask': tokens.attention_mask[i],
        #         'token_type_ids': tokens.token_type_ids[i],
        #     }
        #     self.x.append(token)
        
        # self.y = torch.LongTensor(labelss)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == '__main__':
    print(NERDataset(ds().get_train(), tokenizer))