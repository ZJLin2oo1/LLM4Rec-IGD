#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --gres=gpu:1                    # 分配 1 个 GPU
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00                 # 作业运行时间
#SBATCH --output=log/evaluate.txt

source ~/.bashrc
conda activate IGD
#cd /home/l/linzijie/DecodingMatters

#for category in "Toys_and_Games" ; do
#for category in "Books" ; do
#for category in "Video_Games" ; do
#for category in "CDs_and_Vinyl" ; do

for category in "Books" ; do
    train_file=$(ls -f ./code/train/${category}*11.csv)
    eval_file=$(ls -f ./code/valid/${category}*11.csv)
    test_file=$(ls -f ./code/test/${category}*11.csv)
    info_file=$(ls -f ./code/info/${category}*.txt)
    python ./code/split.py --input_path ${test_file} --output_path ./temp/${category}_base
    cudalist="0"
    for i in ${cudalist}
    do
        echo $i
        CUDA_VISIBLE_DEVICES=$i python -u ./code/evaluate.py \
        --base_model ./output_dir_ig/${category} \
        --train_file ${train_file} --info_file ${info_file} \
        --category ${category} \
        --reference_item_path ./code/item_frequency/${category}.json \
        --test_data_path ./temp/${category}_base/${i}.csv \
        --result_json_data ./temp/${category}_base/${i}.json \
        --alpha 0.0 \
        --length_penalty 0.0  # length_penalty = 1.0 for BIGRec
    done
    wait
    python ./code/merge.py --input_path ./temp/${category}_base --output_path ./output_dir_ig/${category}/final_result.json
    python ./code/calc.py --path ./output_dir_ig/${category}/final_result.json --item_path ${info_file}
done
