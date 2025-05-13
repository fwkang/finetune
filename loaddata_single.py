import pandas as pd
from datasets import Dataset
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
import swanlab
import torch
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch

model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-4B", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)
train_df = pd.read_json('/home/finetune/whole.json')
train_ds = Dataset.from_pandas(train_df)

category = "短信风险类别选项有：是|否"
prompt = '''在这个任务中，你是一位资深的反诈骗网络安全分析师，你的职责是利用你的专业知识和对网络诈骗行为的深刻理解，从短信文本中识别出可能存在的欺诈行为和风险类别。你的工作对于提前预警潜在的网络诈骗，保护用户财产安全和个人信息不被侵犯具有重要意义。现在，请仔细审查以下短信文本，并运用你的专业判断，给出短信的风险类别判断结果。(%s)'''%category

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 2500
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{prompt}<|im_end|>\
        \n<|im_start|>user\n{example['文本']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        truncation=True,  # 防止超长
    )
    response = tokenizer(f"{example['风险类别']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}  

train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
train_dataset.save_to_disk("whole")