import pandas as pd
import json

# 读取 CSV（不修改原文件，跳过无关列）
df = pd.read_csv("/home/finetune/conversation_data_20240306184703.csv", header=None, usecols=[4],skiprows=1)  # 仅读取第3列（索引2）
# print("总列数:", len(df.columns))
# print("前5行数据:\n", df.head())
# 构建 JSON 数据
output_data = [
    {"文本": row[0], "风险类别": "是"}  # 第3列 -> "文本"，固定"风险类别"
    for row in df.itertuples(index=False)
    if "[Local Message]" not in str(row[0])
]

# 保存为 JSON
with open("data_fraud.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

# print("转换完成！输出文件：output.json")