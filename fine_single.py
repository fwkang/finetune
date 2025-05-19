from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from accelerate import Accelerator
from swanlab.integration.huggingface import SwanLabCallback
import swanlab
import os
from transformers import AutoTokenizer
# torchrun --nproc_per_node=8 /home/finetune/fine_single.py
# 初始化 Accelerator
accelerator = Accelerator()
device = accelerator.device

# 加载数据集
train_dataset = load_from_disk("train")

# 加载模型和 Tokenizer
# model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-4B", torch_dtype="auto")
# tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "./Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,  # 根据硬件决定
    trust_remote_code=True,
    device_map="auto"
)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(
    "./Qwen/Qwen3-4B",
    use_fast=False,
    trust_remote_code=True
)

# LoRA 配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# 使用 Accelerator 包装模型和数据集
# model, train_dataset = accelerator.prepare(model, train_dataset)

# 定义训练参数
args = TrainingArguments(
    output_dir="./output/single-class-20epoch",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=2500,
    num_train_epochs=20,
    save_steps=2500,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
)

# SwanLab 日志回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-Finetuning",
    experiment_name="Qwen3-single-class",
    description="Qwen3模型微调。",
    config={
        "model": "./Qwen/Qwen3-4B",
        "dataset": "FGRC-SCD电信诈骗数据集",
    },
    mode="disabled"
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[swanlab_callback],
)

# 开始训练
trainer.train()

# 结束 SwanLab 实验
swanlab.finish()