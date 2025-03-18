#!/bin/bash
source ~/.bashrc

log_file="log/inference.txt"
mkdir -p $(dirname "$log_file")
echo "Starting evaluation..." > $log_file

timestamp=$(date +"%m%d_%H%M")
# 定义变量
#category_range=("Toys_and_Games" "Books" "Video_Games" "CDs_and_Vinyl" "Sports_and_Outdoors")
category_range=("CDs_and_Vinyl")
alpha_range=(0.0)
temperature_range=(1.0)
#alpha_range=(0.0)
#temperature_range=(1.0)

for category in "${category_range[@]}"; do
    train_file=$(ls -f ./code/train/${category}*11.csv)
    eval_file=$(ls -f ./code/valid/${category}*11.csv)
    test_file=$(ls -f ./code/test/${category}*11.csv)
    info_file=$(ls -f ./code/info/${category}*.txt)
    python ./code/split.py --input_path "${test_file}" --output_path "./temp2/${category}_base" >> $log_file 2>&1

    for alpha in "${alpha_range[@]}"; do
        for temperature in "${temperature_range[@]}"; do

            cudalist=("0" "1" "2" "3")  # 可以修改为多个 GPU ID，例如：("0" "1" "2")
            index=0
            for i in "${cudalist[@]}"; do
                CUDA_VISIBLE_DEVICES=$i python ./code/inference.py \
                    --base_model "output_dir_ig/${category}" \
                    --train_file "${train_file}" --info_file "${info_file}" \
                    --category "${category}" \
                    --reference_item_path "./code/item_frequency/${category}.json" \
                    --test_data_path "./temp2/${category}_base/${i}.csv" \
                    --result_json_data ./temp2/${category}_base/${index}.json \
                    --alpha "${alpha}" \
                    --temperature "${temperature}" \
                    --length_penalty 0.0 >> $log_file 2>&1 &
                    ((index++))
            done

            wait

            python ./code/merge.py --input_path ./temp2/${category}_base --output_path ./grid_baseline/${category}/${alpha}_${temperature}.json >> $log_file 2>&1
            python ./code/calc.py --path "./grid_baseline/${category}/${alpha}_${temperature}.json" --item_path "${info_file}" >> $log_file 2>&1

        done
    done

    # ✅ 在所有 alpha 和 temperature 运行完后，调用 my_calc.py 计算 CSV
    python ./code/my_calc.py \
        --input_dir "./grid_baseline/${category}/" \
        --item_path "${info_file}" \
        --output_path "./grid_baseline/${category}/results_${category}_${timestamp}.csv"
done