#!/bin/bash

#SBATCH --job-name=cft
#SBATCH --gres=gpu:1                    # 分配 1 个 GPU
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00                 # 作业运行时间
#SBATCH --output=log/cft.txt

source ~/.bashrc
conda activate D3
#cd /home/l/linzijie/DecodingMatters
#for category in "Toys_and_Games" "Sports_and_Outdoors" "CDs_and_Vinyl" "Video_Games" "Books"

for category in "Toys_and_Games" ; do
    train_file=$(ls -f ./code/train/${category}*11.csv)
    eval_file=$(ls -f ./code/valid/${category}*11.csv)
    test_file=$(ls -f ./code/test/${category}*11.csv)
    info_file=$(ls -f ./code/info/${category}*.txt)
    echo ${train_file} ${test_file} ${info_file} ${eval_file}
#    python -u -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
    python -u code/cft_monitor.py \
        --base_model Qwen/Qwen2.5-1.5B \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --reference_item_path ./code/item_frequency/${category}.json \
        --output_dir ./output_dir_cft/${category} \
        --category ${category} \
        --beta 0.1 \
        --alpha 0.025 \
#    cp Qwen/Qwen2-0.5B*token* ./output_dir/${category}/
done

# beta = 0.09, 0.16, 0.29, 0.38, 0.5, 0.66, 0.9, 0.96
# alpha = 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.3



