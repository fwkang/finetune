from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from accelerate import Accelerator
import swanlab
from transformers import AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback

# 初始化 Accelerator
accelerator = Accelerator()
device = accelerator.device

# 加载并预处理数据集
train_dataset = load_from_disk("train")
def filter_long_samples(examples, max_length=512):
    # 假设输入字段是 "input_ids"
    return [len(ids) <= max_length for ids in examples["input_ids"]]
train_dataset = train_dataset.filter(filter_long_samples, batched=True,batch_size=1000)
test_dataset = load_from_disk("test")
test_dataset = test_dataset.filter(filter_long_samples, batched=True,batch_size=1000)

# 加载模型和 Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "./Qwen/Qwen3-4B",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)

# 增强的LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="lora_only",
)

model = get_peft_model(model, config)
model.config.use_cache = False

# 强制设置只有 LoRA 层为可训练
# def mark_only_lora_as_trainable(model):
#     for name, param in model.named_parameters():
#         if "lora_" in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False

# mark_only_lora_as_trainable(model)
# model.print_trainable_parameters()

# # 设置 use_cache = False 避免 gradient checkpointing 冲突
# model.config.use_cache = False

# # 定义优化器，只取可训练参数
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 强制设置只有 LoRA 层为可训练
# def mark_only_lora_as_trainable(model):
#     for name, param in model.named_parameters():
#         if "lora_" in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False

# mark_only_lora_as_trainable(model)
# model.print_trainable_parameters()

# # 设置 use_cache = False 避免 gradient checkpointing 冲突
# model.config.use_cache = False

# # 定义优化器，只取可训练参数
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# SwanLab 日志回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-Finetuning",
    experiment_name="Qwen3-single-class-optimized",
    description="优化后的Qwen3模型微调，增强数据集影响",
    config={
        "model": "./Qwen/Qwen3-4B",
        "dataset": "FGRC-SCD电信诈骗数据集",
        "lora_rank": 16,
        "learning_rate": 5e-4,
        "epochs": 50,
    },
    mode="disabled"
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
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

