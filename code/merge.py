import fire
import pandas as pd
import json
from tqdm import tqdm
import os

def merge(input_path, output_path, split_num = 1):
    data = []
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for i in tqdm(range(split_num)):
        with open(f'{input_path}/{i}.json', 'r') as f:
            data.extend(json.load(f))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(merge)
