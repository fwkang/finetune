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

# 初始化设置
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-4B", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "./output/single2-class-30epoch/checkpoint-7500/").to("cuda")

# 数据加载
test_df = pd.read_json('eval-400x10.json')

# 二分类配置
categories = ["否", "是"]  # 注意顺序：0=否，1=是
category2index = {v: k for k, v in enumerate(categories)}

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    return response

# 原始标签 -> 二分类
def map_to_binary(label):
    return "是" if label != "无风险" else "否"

# 测试循环
y_true, y_pred = [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    messages = [
        {"role": "system", "content": "请判断以下短信是否存在欺诈风险，直接回答“是”或“否”"},
        {"role": "user", "content": row['文本']}
    ]
    response = predict(messages, model, tokenizer)
    true_label = map_to_binary(row['风险类别'])
    
    # 处理模型响应
    if response in categories:
        pred_label = response
    else:
        pred_label = "是" if any(kw in response for kw in ["是", "存在", "有"]) else "否"
    
    y_true.append(category2index[true_label])
    y_pred.append(category2index[pred_label])

# 评估指标
print("\n评估结果：")
print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
print(f"精确率: {precision_score(y_true, y_pred):.4f}")
print(f"召回率: {recall_score(y_true, y_pred):.4f}")
print(f"F1分数: {f1_score(y_true, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}")

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('预测标签'); plt.ylabel('真实标签'); plt.title('二分类混淆矩阵')
plt.show()