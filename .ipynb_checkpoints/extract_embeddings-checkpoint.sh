#!/bin/bash
#SBATCH --job-name=extract_picme
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/extract_%j.out

module load python/3.9.0
module load cuda/11.8.0

python extract_embeddings.py \
    --model_name baseline \
    --modalities img text_rad text_ds ts demo \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --seed_category single \
    --seed_number 42 \
    --fusion_method concatenation \
    --epochs 1 \
    --objective auroc \
    --metrics auroc \
    --wandb_project extraction_run \
    --wandb_name extraction \
    --task mortality \
    --save_prefix dummy_path \

