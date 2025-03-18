#!/bin/bash

#SBATCH --job-name=extract_rf_item
#SBATCH --gres=gpu:1                    # 分配 1 个 GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00                 # 作业运行时间
#SBATCH --output=log/extract_rf_item.txt

source ~/.bashrc
conda activate D3
cd /home/l/linzijie/DecodingMatters
echo "I am here"
python process_item_frequency.py
#for category in "Toys_and_Games" ; do
#python ./code/extract_rf_item.py --category ${category} --output_folder items_pool
#done
