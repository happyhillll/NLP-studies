# https://nlpinkorean.github.io/illustrated-bert/

import torch
from transformers import AutoTokenizer, AutoModel

# https://github.com/huggingface/transformers/tree/main/examples/pytorch
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py

plm = "klue/roberta-base"   # https://huggingface.co/models
tokenizer = AutoTokenizer.from_pretrained(plm)


xs = ["안녕하세요.", "아이들의 행복 응원하고 해피니스링 받기"]
tokens_wo_pad = tokenizer(xs)      # https://huggingface.co/docs/transformers/main_classes/tokenizer
tokens = tokenizer(xs, max_length=512, truncation=True, padding="max_length")

model = AutoModel.from_pretrained(plm)

input_ids = torch.LongTensor(tokens.input_ids)              # (2, 512)
attention_mask = torch.LongTensor(tokens.attention_mask)    # (2, 512)
token_type_ids = torch.LongTensor(tokens.token_type_ids)    # (2, 512)

# https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/roberta#transformers.RobertaModel.forward
output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids
)
# output.last_hidden_state  # (2, 512, 768)
# output.pooler_output      # (2, 768)



#  https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py