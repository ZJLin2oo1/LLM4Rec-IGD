#!/bin/bash

#SBATCH --job-name=extract_rf_item
#SBATCH --gres=gpu:1                    # 分配 1 个 GPU
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00                 # 作业运行时间
#SBATCH --output=log/extract_rf_item.txt

source ~/.bashrc
conda activate D3
cd /home/l/linzijie/DecodingMatters
python compute_item_frequency.py \
  --meta_file code/info/Video_Games_5_2012-10-2018-11.txt \
  --train_file code/train/Video_Games_5_2012-10-2018-11.csv \
  --output_file code/item_frequency/Video_Games.json
