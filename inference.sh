#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --gres=gpu:1                    # 分配 1 个 GPU
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00                 # 作业运行时间
#SBATCH --output=log/inference.txt

source ~/.bashrc
conda activate D3
cd /home/l/linzijie/DecodingMatters

timestamp=$(date +"%m%d_%H%M")
# 定义变量
#category_range=("Toys_and_Games" "Books" "Video_Games" "CDs_and_Vinyl" "Sports_and_Outdoors")
category_range=("Books")
alpha_range=(0.0 0.1)
temperature_range=(1.0)
#alpha_range=(0.0)
#temperature_range=(1.0)

for category in "${category_range[@]}"; do
    train_file=$(ls -f ./code/train/${category}*11.csv)
    eval_file=$(ls -f ./code/valid/${category}*11.csv)
    test_file=$(ls -f ./code/test/${category}*11.csv)
    info_file=$(ls -f ./code/info/${category}*.txt)
#    python ./code/split.py --input_path "${test_file}" --output_path "./temp/${category}_base"

    for alpha in "${alpha_range[@]}"; do
        for temperature in "${temperature_range[@]}"; do

            cudalist=("0")  # 可以修改为多个 GPU ID，例如：("0" "1" "2")
            for i in "${cudalist[@]}"; do
                echo "Using GPU: $i"
                CUDA_VISIBLE_DEVICES=$i python ./code/inference.py \
                    --base_model "./output_dir_ig/${category}" \
                    --train_file "${train_file}" --info_file "${info_file}" \
                    --category "${category}" \
                    --reference_item_path "./code/item_frequency/${category}.json" \
                    --test_data_path "./temp/${category}_base/${i}.csv" \
                    --result_json_data "./grid_baseline/${category}/${alpha}_${temperature}.json" \
                    --alpha "${alpha}" \
                    --temperature "${temperature}" \
                    --length_penalty 0.0
            done

            wait

            python ./code/calc.py --path "./grid_baseline/${category}/${alpha}_${temperature}.json" --item_path "${info_file}"

        done
    done

    # ✅ 在所有 alpha 和 temperature 运行完后，调用 my_calc.py 计算 CSV
    python ./code/my_calc.py \
        --input_dir "./grid_baseline/${category}/" \
        --item_path "${info_file}" \
        --output_path "./grid_baseline/${category}/results_${category}_${timestamp}.csv"
done


#                    --base_model "./models/BIGRec_${category}_0.5B" \

