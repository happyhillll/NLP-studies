from transformers import AutoTokenizer
from nerdata import *
from torch.utils.data import Dataset
# tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
# MAX_TOKEN_LEN = 512

# label_set = ["O", "B-Person", "I-Person", "[PAD]"]


# sentences = [["마크크크", "주커버그는", "밥을", "먹었다"], ["허철훈은", "슬프다"]]
# labels = [["B-Person", "I-Person", "O", "O"], ["B-Person", "O"]]

# label_to_ids = {l:i for i, l in enumerate(label_set)}

class dataset(Dataset):
    def __init__(self,data,tokenizer):
        self.preprocess(data,tokenizer)
    
    def preprocess(self,data,tokenizer):
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id
        MAX_TOKEN_LEN = 512

        sentences = ds().get_train()[0][:1000]
        texts = [text.split(' ') for text in sentences]
        labels = ds().get_train()[1][:1000]
        labels = [l.split(' ') for l in labels]

        label_set = ds().get_label()

        label_to_ids = {l:i for i, l in enumerate(label_set)}
        label_to_ids['[PAD]'] = 30
        label_to_ids['[CLS]'] = 31
        label_to_ids['[SEP]'] = 32
        label_to_ids['[UNK]'] = 33

        all_input_ids=[]
        all_attention_mask=[]
        all_token_type_ids=[]

        for sentence, label in zip(texts, labels):
            sent_tokens=[]
            sent_labels=[]  
            for w in sentence:
                word_tokens=tokenizer.tokenize(w)
                sent_tokens.extend(word_tokens)
            for l in label:
                sent_labels.extend([l]+[unk_token]*(len(word_tokens)-1))
            
            #label_ids
            label_ids=[]
            for label in sent_labels:
                idx=label_to_ids[label]
                label_ids.append(idx)
            
            sent_tokens=[cls_token]+sent_tokens+[sep_token]
            sent_labels=[pad_token_id]+label_ids+[pad_token_id]
            
            #sent_tokens랑 sent_labels 512길이로 패딩
            sent_tokens.extend([pad_token_id]*(MAX_TOKEN_LEN-len(sent_tokens)))
            sent_labels.extend([pad_token_id]*(MAX_TOKEN_LEN-len(sent_labels)))
            
            sent_tokens = [str(token) for token in sent_tokens]
            
            input_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
            attention_mask = [1 if token_id != 1 else 0 for token_id in input_ids] #패딩토큰에는 0
            token_type_ids = [0 if token_id == 1 else 1 for token_id in attention_mask]
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)
        return all_input_ids, all_attention_mask, all_token_type_ids, sent_labels
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]