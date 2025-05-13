import json
import os

# 假设你要读取的JSON文件列表
input_files = ["/home/finetune/eval-400x10.json", "/home/finetune/multi-class-1000x10.json"]  # 替换为你的实际文件名

output_file = "/home/finetune/whole.json"

all_data = []

# 遍历所有输入文件
for file in input_files:
    if os.path.exists(file):  # 检查文件是否存在
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # 修改“风险类别”字段
                risk_category = item["风险类别"]
                item["风险类别"] = "否" if risk_category == "无风险" else "是"
                all_data.append(item)
    else:
        print(f"文件 {file} 不存在，跳过。")

with open("/home/finetune/data_normal.json",'r',encoding='utf-8') as f:
    data1=json.load(f)

all_data+=data1

# 将处理后的数据写入新文件 good.json
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)



print(f"处理完成，结果已保存至 {output_file}")