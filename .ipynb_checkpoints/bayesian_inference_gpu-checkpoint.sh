#!/bin/bash
#SBATCH --job-name=mnar_gpu_sweep
#SBATCH --array=1-12             
#SBATCH --time=24:00:00             
#SBATCH --mem=128G               
#SBATCH --cpus-per-task=4
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1           
#SBATCH --output=logs/gpu_sweep_%a.out

module purge
unset LD_LIBRARY_PATH
module load cuda cudnn

scales=(0.01 1.0 10.0)

TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))

SCALE_INDEX=$((TASK_ID / 4))
CURRENT_SCALE=${scales[$SCALE_INDEX]}

CHAIN_ID=$((TASK_ID % 4 + 1))
python bayesian_inference_gpu.py $CHAIN_ID $CURRENT_SCALE