from transformers import AutoTokenizer
from nerdata import *

# tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
# MAX_TOKEN_LEN = 512

# label_set = ["O", "B-Person", "I-Person", "[PAD]"]


# sentences = [["마크크크", "주커버그는", "밥을", "먹었다"], ["허철훈은", "슬프다"]]
# labels = [["B-Person", "I-Person", "O", "O"], ["B-Person", "O"]]

# label_to_ids = {l:i for i, l in enumerate(label_set)}

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
MAX_TOKEN_LEN = 512

sentences = ds().get_train()[0][:1000]
texts = [text.split(' ') for text in sentences]
#이중리스트 없애기
flat_texts=sum(texts,[])
labels = ds().get_train()[1][:1000]
labels = [l.split(' ') for l in labels]
#이중리스트 없애기
flat_labels=sum(labels,[])

label_set = ds().get_label()

label_to_ids = {l:i for i, l in enumerate(label_set)}

label_id=[]
label_ids=[]
tokens=[]
tokenized_word=[]
int_label=[]
sentence_label_ids=[]

# for sentence, label in zip(texts, labels):
#     word_tokens = []
#     for i in sentence:
#         word_tokens=tokenizer.tokenize(i)
#         tokens.append(word_tokens)
#         for l in label:
#             label_id=label_to_ids[l]
#             label_ids.extend([label_id]+['UNK']*(len(word_tokens)-1))

label_idx=[]
#label_to_ids
for i in flat_labels:
    id=label_to_ids.get(i)
    label_ids.append(id)

for sentence, label in zip(texts, label_ids):
    word_tokens = []
    for i in sentence:
        word_tokens=tokenizer.tokenize(i)
        tokens.append(word_tokens)
        label_idx.append([label]+['UNK']*(len(word_tokens)-1))
    #label이 변수라서 안돌아감..
    #차라리 label 전부 ids로 바꿔버리고 넣는게 나을듯 
        
    # for sentence, label in zip(texts, labels):
#     tokenized_sentence=[]
#     for word in sentence:
#         tokenized_word=tokenizer.tokenize(word)
#         tokenized_sentence.extend(tokenized_word)
#     tokens.append(tokenized_sentence)
#     for i in label:
#         sentence_label_ids = [label_to_ids[label] for label in i]
#     label_ids.append(sentence_label_ids)
        
    #     int_label=label_to_ids.get(i)
    #     sentence_label_ids.extend(int_label)
    # label_ids.append(sentence_label_ids)
    #  tokens.append(tokenized_sentence)
    # label_ids.append(label_to_ids.get(label))
# 분명 zip으로 푸는 방법이 있을텐데. 일단 보류?    
    


for word,label in zip(texts,labels):
    for sentence in texts:
        tokenized_sentence=[]
        for word in sentence:
            tokenized_sentence += tokenizer.tokenize(word)
        tokens.append(tokenized_sentence)
    for l in label:
        label_ids.append(label_to_ids.get(l))
    
    
    
    
    # wordd.append(word)
    # label_ids.append(label)
    
    # for i in word:
    #     tokenized_word = tokenizer.tokenize(i)
    #     wordd.append(tokenized_word)
        
        
    label_ids.extend([label_to_ids.get(label)])+['UNK']*(len(word))
    
    label_ids.extend([int(label) for label in slot_label.split(',')] + [unk_token] * (len(word_tokens) - 1))

tokens=[]
for words in texts:
    for w in words:
        word_tokens=tokenizer.tokenize(w)
        tokens+=word_tokens
        print()