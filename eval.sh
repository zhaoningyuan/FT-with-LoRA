#!/bin/sh

# 设置模型路径前缀
peft_model_path_prefix="qwen1.5-7b-prompt"

# 创建一个空数组来存储结果
checkpoints=()

# 遍历目录下的所有文件和文件夹
for item in "$peft_model_path_prefix"/*; do
  # 检查是否是以 'checkpoint' 开头的文件夹
  if [[ $(basename "$item") == checkpoint* && -d "$item" ]]; then
    # 如果是，添加到数组中
    checkpoints+=("$item")
  fi
done

for checkpoint in "${checkpoints[@]}"; do
  python evaluate.py \
    --model_name_or_path "/models/Qwen1.5-7B" \
    --peft_model_path $checkpoint \
    --test_file "./data/pCLUE_test_public.csv" \
    --output_file "result.json" \
    --num_samples 300 \
    --device "cuda:1" 
  
done