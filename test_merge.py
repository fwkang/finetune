import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib import rcParams
import shutil

# 设置中文字体和图像显示
rcParams['font.family'] = ['sans-serif']
rcParams['font.sans-serif'] = ['SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

# ==== 路径定义 ====
base_model_path = "./Qwen/Qwen3-4B"
adapter_path = "./output/single2-class-30epoch/checkpoint-2370/"
untied_path = "./untied_model"
merged_path = "./merged_model"

# ==== Step 1: 加载 tokenizer（始终加载） ====
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)

# ==== Step 2: 解绑 base 模型并保存 ====
print("加载基础模型（解绑权重）...")
if not os.path.exists(untied_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, tie_word_embeddings=False, torch_dtype="auto"
    )
    # 手动复制 lm_head 的参数
    base_model.lm_head.weight.data = base_model.model.embed_tokens.weight.data.clone()

    print(f"保存解绑模型到：{untied_path}")
    base_model.save_pretrained(untied_path)
    tokenizer.save_pretrained(untied_path)
else:
    print("解绑模型已存在，跳过保存。")

# ==== Step 3: 加载解绑模型 + 合并 Adapter 并保存 ====
print("加载 Adapter 并合并权重...")
base_model = AutoModelForCausalLM.from_pretrained(untied_path, torch_dtype="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()
model.eval()

# 如果目录已存在，先删除
if os.path.exists(merged_path):
    shutil.rmtree(merged_path)

print(f"保存合并后的模型到：{merged_path}")
model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

# ==== Step 4: 重新加载合并模型用于推理 ====
print("重新加载合并模型用于部署...")
model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype="auto").to("cuda")
tokenizer = AutoTokenizer.from_pretrained(merged_path, use_fast=False, trust_remote_code=True)
model.eval()

# ==== Step 5: 加载评估数据 ====
test_df = pd.read_json('eval-400x10.json')

# ==== Step 6: 定义辅助函数 ====
categories = ["否", "是"]
category2index = {v: k for k, v in enumerate(categories)}

def map_to_binary(label):
    return "是" if label != "无风险" else "否"

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    return response

# ==== Step 7: 执行预测 ====
y_true, y_pred = [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    messages = [
        {"role": "system", "content": "请判断以下短信是否存在欺诈风险，直接回答“是”或“否”"},
        {"role": "user", "content": row['文本']}
    ]
    response = predict(messages, model, tokenizer)
    print(f"文本: {row['文本']}")
    print(f"模型输出: {response}")
    true_label = map_to_binary(row['风险类别'])

    # 处理模型响应的非标准格式
    if response in categories:
        pred_label = response
    else:
        pred_label = "是" if any(kw in response for kw in ["是", "存在", "有"]) else "否"
    
    y_true.append(category2index[true_label])
    y_pred.append(category2index[pred_label])

# ==== Step 8: 打印评估结果 ====
print("\n评估结果：")
print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
print(f"精确率: {precision_score(y_true, y_pred):.4f}")
print(f"召回率: {recall_score(y_true, y_pred):.4f}")
print(f"F1分数: {f1_score(y_true, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}")

# ==== Step 9: 绘制混淆矩阵 ====
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('二分类混淆矩阵')
plt.show()
