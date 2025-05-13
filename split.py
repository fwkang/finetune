import json
import random

def split_data(data, train_ratio=0.7, test_ratio=0.2):
    total = len(data)
    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)

    return data[:train_end], data[train_end:test_end], data[test_end:]

def main():
    input_file = "/home/finetune/whole.json"
    train_file = "/home/finetune/data/train.json"
    test_file = "/home/finetune/data/test.json"
    eval_file = "/home/finetune/data/eval.json"

    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 打乱顺序
    random.shuffle(data)

    # 划分数据集
    train_data, test_data, eval_data = split_data(data)

    # 写入文件
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)

    print(f"处理完成，共 {len(data)} 条数据。")
    print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}, 验证集: {len(eval_data)}")

if __name__ == "__main__":
    main()