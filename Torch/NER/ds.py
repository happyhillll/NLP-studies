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
labels = ds().get_train()[1][:1000]
labels = [label.split(' ') for label in labels]

label_set = ds().get_label()

label_to_ids = {l:i for i, l in enumerate(label_set)}

label_ids=[]
word=[]
tokens=[]

for sentence, label in zip(texts, labels):
    tokenized_sentence=[]
    for i in sentence:
       tokenized_word = tokenizer.tokenize(i)
       tokenized_sentence.append(tokenized_word)
    tokens.append(tokenized_sentence)
    label_ids.append(label_to_ids.get(label))
        
#         tokenized_sentence=tokenizer.tokenize(i)
#         tokens.append(tokenized_sentence)
       
       

# label_ids.append(label_to_ids.get(label))
# 분명 zip으로 푸는 방법이 있을텐데. 일단 보류?    



# for word,label in zip(texts,labels):
#     for sentence in texts:
#         tokenized_sentence=[]
#         for word in sentence:
#             tokenized_sentence += tokenizer.tokenize(word)
#         tokens.append(tokenized_sentence)
#     for l in label:
#         label_ids.append(label_to_ids.get(l))
    
    
    
    
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