import os
from enum import Enum

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, PromptTuningConfig, TaskType, PromptTuningInit, PrefixTuningConfig, LNTuningConfig
from evaluate_pclue import normalize, f1_sim, rouge_l_zh


DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

def preprocess(text):
    return text.replace("\n", "_")
def postprocess(text):
    return text.replace("_", "\n")

def answer_fn(model_trained, tokenizer, text, sample=False, top_p=0.6):
    '''sample：是否抽样。生成任务，可以设置为True;
        top_p：0-1之间，生成的内容越多样、
    '''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(model_trained.device) 
    input_length = encoding["input_ids"].size(1)
    if not sample: # 不进行采样
        out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=128, num_beams=4, length_penalty=0.6)
    else: # 采样（生成）
        out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=128, do_sample=True, top_p=top_p)
    out_text = tokenizer.batch_decode(out["sequences"][:, input_length:], skip_special_tokens=True)
    return postprocess(out_text[0])  

def evaluate_pclue_fn(predict_answer, target_answer, type_):
    if isinstance(target_answer, list):  # 将列表转换为字符串，如关键词生成
        target_answer = "，".join(target_answer)
    target_answer=normalize(target_answer)
    predict_answer=normalize(predict_answer)
    if len(predict_answer)==0:
        predict_answer = "无"
    if type_=='classify' or type_=='anaphora_resolution': # 分类
        label_temp=True if target_answer==predict_answer else False 
        return label_temp
    elif type_=='mrc': # 阅读理解
        em=1 if target_answer==predict_answer else 0
        f1=f1_sim(predict_answer,target_answer)
        return (em, f1)
    elif type_=='generate': # 生成
        rouge_l=rouge_l_zh(target_answer, predict_answer)
        return rouge_l
    elif type_=='nli': # 推理
        label_temp = True if target_answer == predict_answer else False
        return label_temp
    else:
        print("error...predict_line:",predict_answer,";target_line:",target_answer)
        return None

    
def my_create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    text_column = "input"
    label_column = "target"
    max_length = data_args.max_seq_length
    def tokenize_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
        # input + label + EOS
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        # padding
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    dataset = load_dataset("csv", data_files={"train": data_args.train_dataset_path, "valid": data_args.valid_dataset_path})
    dataset["train"] = dataset["train"].select(range(data_args.train_examples_num))
    dataset["valid"] = dataset["valid"].select(range(data_args.valid_examples_num))
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=dataset["train"].column_names,load_from_cache_file=False,desc="Running tokenizer on dataset",)
    # tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1,load_from_cache_file=False,desc="Running tokenizer on dataset",)
    train_data = tokenized_datasets["train"]
    valid_data = tokenized_datasets["valid"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0].keys()}")
    print(f"A sample of dev dataset: {valid_data[0].keys()}")
    return train_data, valid_data


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )
        # peft_config = PromptTuningConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     prompt_tuning_init=PromptTuningInit.TEXT,
        #     num_virtual_tokens=8,
        #     prompt_tuning_init_text="Read the context below and answer the associated question.",
        #     tokenizer_name_or_path=args.model_name_or_path,
        # )
        # peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
        # peft_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM,)


    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer
