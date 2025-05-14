from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from accelerate import Accelerator
from swanlab.integration.huggingface import SwanLabCallback
import swanlab
import os
from transformers import AutoTokenizer

# 初始化 Accelerator
accelerator = Accelerator()
device = accelerator.device

# 加载数据集
train_dataset = load_from_disk("train")
test_dataset=load_from_disk("test")

# 加载模型和 Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "./Qwen/Qwen3-4B", 
    torch_dtype="auto",
    # device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)

# 增强的LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    inference_mode=False,
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="lora_only",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# 定义优化的训练参数
args = TrainingArguments(
    output_dir="./output/single3-class-30epoch",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=500,
    num_train_epochs=30,
    save_steps=1000,
    learning_rate=2e-4,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    fp16=True,

    eval_strategy="steps",            # 每隔一定步数进行评估
    save_strategy="steps",                  # 与上面一致
    eval_steps=500,                         # 每 500 步评估一次
    metric_for_best_model="loss",           # 使用 loss 判断最佳模型
    load_best_model_at_end=True,            # 训练结束加载最优模型
    greater_is_better=False,                # loss 越小越好
)

# SwanLab 日志回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-Finetuning",
    experiment_name="Qwen3-single-class-optimized",
    description="优化后的Qwen3模型微调，增强数据集影响",
    config={
        "model": "./Qwen/Qwen3-4B",
        "dataset": "FGRC-SCD电信诈骗数据集",
        "lora_rank": 16,
        "learning_rate": 2e-4,
        "epochs": 30,
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
    optimizers=(torch.optim.AdamW(model.parameters(), lr=2e-4), None),
    eval_dataset=test_dataset,
)

# 开始训练
trainer.train()

# 结束 SwanLab 实验
swanlab.finish()