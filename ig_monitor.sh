#!/bin/bash

#SBATCH --job-name=train
#SBATCH --gres=gpu:1                    # 分配 1 个 GPU
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00                 # 作业运行时间
#SBATCH --output=log/ig_monitor.txt

source ~/.bashrc
conda activate D3

# Toys_and_Games: beta=0.5
# CDs_and_Vinyl: beta=0.1
# Video_Games: beta=0.2
# Books: beta=0.1

#for category in "Toys_and_Games" "CDs_and_Vinyl" "Video_Games" "Books"
for category in "Video_Games" ; do
    train_file=$(ls -f ./code/train/${category}*11.csv)
    eval_file=$(ls -f ./code/valid/${category}*11.csv)
    test_file=$(ls -f ./code/test/${category}*11.csv)
    info_file=$(ls -f ./code/info/${category}*.txt)
    echo ${train_file} ${test_file} ${info_file} ${eval_file}
#    python -u -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
    python -u code/ig_monitor.py \
        --base_model Qwen/Qwen2.5-1.5B \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --reference_item_path ./code/item_frequency/${category}.json \
        --output_dir ./output_dir_ig/${category} \
        --category ${category} \
        --beta 0.1 
done



