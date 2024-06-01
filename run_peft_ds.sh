accelerate launch --config_file "configs/deepspeed_config.yaml"  train.py \
--seed 100 \
--model_name_or_path "/models/Qwen1.5-7B" \
--train_dataset_path "./data/pCLUE_train.csv" \
--valid_dataset_path "./data/pCLUE_dev.csv" \
--train_examples_num 10000 \
--valid_examples_num 1000 \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 1024 \
--num_train_epochs 5 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "qwen1.5-7b-prompt" \
--per_device_train_batch_size 3 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 1 \
--gradient_checkpointing True \
--lora_target_modules "q_proj,k_proj" \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--use_reentrant False \
--dataset_text_field "input" \
--use_peft_lora True \
--use_flash_attn True \
--use_4bit_quantization False
# --push_to_hub \
# --hub_private_repo True \
# --hub_strategy "every_save" \
# --dataset_name "smangrul/ultrachat-10k-chatml" \
# --chat_template_format "chatml" \