#토크나이저 불러오기
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

# 파인튜닝한 모델의 체크포인트 로딩하기
import torch
fine_tuned_model_ckpt=torch.load(args.downstream_model_checkpoint_fpath, map_location=torch.device("cpu"))

#저장한 모델의 BERT 설정 로드
from transformers import BertConfig
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)

#해당 설정값 적용하기
from transformers import BertForTokenClassification
model = BertForTokenClassification(pretrained_model_config)

#BERT모델의 체크포인트 읽기 ? 
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

#모델 평가모드
model.eval()

________

#input 토크나이저에 넣고
# input_ids, attention_mask, token_type_ids 만들어
# 파이토치 텐서로 변환해서 모델에 입력해

def inference(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        probs = outputs.logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[pred.item()] for pred in preds]
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                token_result = {
                    "token": token,
                    "predicted_tag": predicted_tag,
                    "top_prob": str(round(top_prob[0].item(), 4)),
                }
                result.append(token_result)
    return {
        "sentence": sentence,
        "result": result,
    }