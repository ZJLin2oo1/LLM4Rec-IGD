#!/bin/bash

#SBATCH --job-name=compute_rf_item
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=log/item_freq.txt

source ~/.bashrc
conda activate IGD

for category in "Toys_and_Games" "CDs_and_Vinyl" "Video_Games" "Books"; do
    # preprocess 
    python ./code/preprocess.py --category "$category"

    # compute item-freq
    meta_file=$(ls ./code/info/${category}_5_*.txt | head -n1)
    train_file=$(ls ./code/train/${category}_5_*.csv | head -n1)
    output_file=./code/item_frequency/${category}.json

    echo "[INFO] Processing category: $category"
    echo "       Meta file:   $meta_file"
    echo "       Train file:  $train_file"
    echo "       Output file: $output_file"

    if [[ ! -f "$meta_file" || ! -f "$train_file" ]]; then
        echo "[WARNING] Skipping $category: file not found."
        continue
    fi

    python ./code/compute_item_freq.py \
      --meta_file "$meta_file" \
      --train_file "$train_file" \
      --output_file "$output_file"
done