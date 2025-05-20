from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import swanlab
from transformers import AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback

# 加载数据集
train_dataset = load_from_disk("train")
test_dataset = load_from_disk("test")

# 加载模型和 Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "./Qwen/Qwen3-4B", 
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)

# LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="lora_only",
)
model = get_peft_model(model, config)

# 只训练LoRA层
def mark_only_lora_as_trainable(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
mark_only_lora_as_trainable(model)
model.print_trainable_parameters()

# 为避免 gradient checkpointing 冲突
model.config.use_cache = False

# 定义优化器
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

# 设置训练参数
args = TrainingArguments(
    output_dir="./output/single3-class-30epoch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=500,
    num_train_epochs=30,
    save_steps=1000,
    learning_rate=2e-4,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    save_on_each_node=True,
    report_to="none",
    remove_unused_columns=False,
    fp16=True,  # 设置为 True，如果你是 A100/H100 也可以设置 bf16=True
    # bf16=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    metric_for_best_model="loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    # ✅ 关键参数：支持 DDP 多卡训练避免因未使用参数报错
    ddp_find_unused_parameters=False,
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
    optimizers=(optimizer, None),
    eval_dataset=test_dataset,
)

# 开始训练
trainer.train()

# 结束 SwanLab 实验
swanlab.finish()
