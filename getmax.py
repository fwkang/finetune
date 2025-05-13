import pandas as pd

# 读取 JSON 文件为 DataFrame
df = pd.read_json('/home/finetune/whole.json')

# 找到"文本"列中最长字符串的第一个出现位置
max_len_index = df['文本'].str.len().idxmax()

# 获取最长文本
longest_text = df.loc[max_len_index, '文本']

print("最长文本是：", len(longest_text))