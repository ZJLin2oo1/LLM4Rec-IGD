import csv
import json
from collections import Counter


def load_item_titles(input_file):
    """加载 item_id 和 item_title 的映射"""
    item_dict = {}

    with open(input_file, 'r', encoding='utf-8') as file:
        items = file.readlines()

        # 处理每一行，最后部分是 item_id，前面部分是 item_title
        for line in items:
            parts = line.strip().split('\t')  # 按 `\t` 分割
            item_title = '\t'.join(parts[:-1])  # 前面部分是 item_title
            item_id = int(parts[-1].strip())  # 最后一部分是 item_id，转换为整数

            # 将 item_id 映射到对应的 item_title
            item_dict[item_id] = item_title

    return item_dict


def count_item_frequencies(train_file, item_dict):
    """计算训练数据中每个 item_id 的频率"""
    item_id_counter = Counter()

    with open(train_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            item_id = int(row['item_id'])  # 获取 item_id
            item_id_counter[item_id] += 1  # 统计频率

    # 通过 item_id 查找对应的 item_title，并记录频率
    item_frequency_dict = {}
    for item_id, frequency in item_id_counter.items():
        item_title = item_dict.get(item_id, "Unknown Item")  # 获取 item_title，如果没找到则用 "Unknown Item"
        item_frequency_dict[item_title] = frequency

    return item_frequency_dict


def save_item_frequencies_to_json(output_file, item_frequency_dict):
    """将 item_title 和 frequency 保存到 JSON 文件"""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(item_frequency_dict, file, ensure_ascii=False, indent=4)

    print(f"统计完成，结果已保存至 {output_file}")
    print(f"统计完成，结果已保存至 {output_file}")


# 使用示例
input_file = 'code/info/Video_Games_5_2012-10-2018-11.txt'  # 存储 item_title 和 item_id 的文件
train_file = 'code/train/Video_Games_5_2012-10-2018-11.csv'  # 训练数据文件，包含 item_id
output_file = 'code/item_frequency/Video_Games.json'  # 保存 item_title 和 frequency 的结果文件

# 载入 item_id 和 item_title 的字典
item_dict = load_item_titles(input_file)

# 获取 item_title 和 frequency pair
item_frequency_dict = count_item_frequencies(train_file, item_dict)

# 保存到 JSON 文件
save_item_frequencies_to_json(output_file, item_frequency_dict)