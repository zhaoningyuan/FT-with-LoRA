import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import create_and_prepare_model, create_datasets, my_create_datasets, answer_fn, evaluate_pclue_fn
from datasets import load_dataset
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="/models/Qwen1.5-7B")
parser.add_argument("--peft_model_path", type=str, default="qwen1.5-7b-lora")
parser.add_argument("--test_file", type=str, default="data/pCLUE_test_public.csv")
parser.add_argument("--output_file", type=str, default="result.json")
parser.add_argument("--num_samples", type=int, default=3000)
parser.add_argument("--device", type=str, default="cuda:1")
args = parser.parse_args()


model_name_or_path = args.model_name_or_path
peft_model_path = args.peft_model_path


test_file = args.test_file
dataset = load_dataset("csv", data_files={"test": test_file})
test_ds = dataset["test"].select(range(args.num_samples))
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.bfloat16, )
model.load_adapter(peft_model_path)
model.to(args.device)


classify_list = []
mrc_list = []
generate_list = []
nli_list = []
from tqdm import tqdm
result = []
with torch.no_grad():
    for i, data in tqdm(enumerate(test_ds)):
        text = data["input"]
        output = answer_fn(model, tokenizer, text=text)
        target_answer = data["target"]
        predict_answer = output.strip()
        # # FIXME: fix for prompt tuning
        # if ": " in predict_answer[: 10]:
        #     predict_answer = predict_answer[predict_answer.index(": ") + 2:]
        # if "：" in predict_answer[: 10]:
        #     predict_answer = predict_answer[predict_answer.index("：") + 1:]
        type_ = data["type"]
        score = evaluate_pclue_fn(predict_answer, target_answer, type_)
        if type_=='classify' or type_=='anaphora_resolution': # 分类
            classify_list.append(score)
        elif type_=='mrc': # 阅读理解
            mrc_list.append(score)
        elif type_=='generate': # 生成
            generate_list.append(score)
        elif type_=='nli': # 推理
            nli_list.append(score)
        else:
            raise ValueError(f"Unknown type: {type_}")
        if i<10: 
            print(i, 'target_answer:',target_answer,";predict_answer:",predict_answer) # 显示部分内容
        result.append({"target": target_answer, "predict": predict_answer, "type": type_, "score": score})
# 计算总分
classify_score=np.average(classify_list)
nli_score=np.average(nli_list) 
generate_score=np.average(generate_list)
mrc_em_score=np.average([x[0] for x in mrc_list])
mrc_f1_score=np.average([x[1] for x in mrc_list])
mrc_score=np.average([mrc_em_score,mrc_f1_score])
# 计算总分
score=np.average([classify_score,nli_score,generate_score,mrc_score])
# 保存分数
result_dict={"score":score,"classify_score":classify_score,"nli_score":nli_score,"generate_score":generate_score,
            "mrc_em_score":mrc_em_score,"mrc_f1_score":mrc_f1_score}
result.append(result_dict)
import json
with open(os.path.join(args.peft_model_path, args.output_file), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
