import os
import fire
import math
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


def gao(input_dir, item_path, output_path):
    """
    计算 HR 和 NDCG，并保存到 CSV（不包含 Category 列）。
    """
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]

    # 读取 item 信息
    with open(f"{item_path}.txt", 'r') as f:
        items = f.readlines()
    item_names = ['\t'.join(_.split('\t')[:-1]).strip().strip("\"") for _ in items]
    item_dict = {name: idx for idx, name in enumerate(item_names)}

    topk_list = [1, 3, 5, 10, 20]
    results = []

    # 遍历 output_dir/{category}/ 目录下所有 JSON 结果文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)

            # 从文件名提取 alpha 和 temperature
            filename_without_ext = os.path.splitext(filename)[0]  # 去掉 .json
            parts = filename_without_ext.split("_")  # 按 "_" 拆分

            if len(parts) == 2:  # 确保拆分后有两个元素
                alpha = float(parts[0])
                temperature = float(parts[1])
                print(f"Extracted alpha: {alpha}, temperature: {temperature}")
            else:
                raise ValueError(f"Invalid filename format: {filename}")

            # 读取 JSON 结果
            with open(file_path, 'r') as f:
                test_data = json.load(f)

            ALLNDCG = np.zeros(5)
            ALLHR = np.zeros(5)

            for sample in tqdm(test_data, desc=f"Processing {filename}"):
                predictions = [_.strip() for _ in sample["predict"]]
                target_item = sample['output'][0].strip() if isinstance(sample['output'], list) else sample[
                    'output'].strip()

                minID = float("inf")
                for i, pred in enumerate(predictions):
                    if pred == target_item:
                        minID = i
                        break

                for index, topk in enumerate(topk_list):
                    if minID < topk:
                        ALLNDCG[index] += 1 / math.log2(minID + 2)
                        ALLHR[index] += 1

            num_samples = len(test_data)
            results.append([
                alpha, temperature,
                *(ALLHR / num_samples),  # HR1, HR3, HR5, HR10, HR20
                *(ALLNDCG / num_samples / (1.0 / math.log2(2)))  # NDCG1, NDCG3, NDCG5, NDCG10, NDCG20
            ])

    # 先按 alpha 排序，再按 temperature 排序
    results.sort(key=lambda x: (x[0], x[1]))

    # 创建 DataFrame（不含 Category 列）
    columns = ["Alpha", "Temperature", "HR1", "HR3", "HR5", "HR10", "HR20",
               "NDCG1", "NDCG3", "NDCG5", "NDCG10", "NDCG20"]

    df = pd.DataFrame(results, columns=columns)

    df = df.round(8)
    # ✅ 直接覆盖写入 (`mode='w'`)，确保 CSV 是有序的
    df.to_csv(output_path, mode='w', header=True, index=False)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    fire.Fire(gao)
