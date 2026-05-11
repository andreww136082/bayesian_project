#!/bin/bash

# ======================= SLURM JOB ARRAY DIRECTIVES =======================
#SBATCH -p 3090-gcondo --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 8 CPU core(s)
#SBATCH -n 2

# Memory
#SBATCH --mem=50G
#SBATCH -t 10:00:00

# Provide a job name and logging based on array ID
#SBATCH -J finetune_sweep_%A_%a
#SBATCH -o logs/finetune_sweep_%A_%a.out
#SBATCH -e logs/finetune_sweep_%A_%a.err
# ==========================================================================

module load python/3.9.16s-x3wdtvt
# <source your environment>

# --- 3D HYPERPARAMETER GRID ---
LEARNING_RATES=(0.000005 0.000001 0.00005 0.00001 0.0005 0.0001)
FUSION_ARRAY=("concatenation" "vanilla_lstm")
INIT_ARRAY=("baseline" "contrastive")

# Calculate indices based on the SLURM_ARRAY_TASK_ID
LR_IDX=$((SLURM_ARRAY_TASK_ID % 6))
FUSION_IDX=$(((SLURM_ARRAY_TASK_ID / 6) % 2))
INIT_IDX=$((SLURM_ARRAY_TASK_ID / 12))

LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
FUSION_METHOD=${FUSION_ARRAY[$FUSION_IDX]}
INIT_MODE=${INIT_ARRAY[$INIT_IDX]}

# --- STATIC PARAMETERS ---
MODALITIES=("img" "ts" "demo" "text_rad" "text_ds")
SEED_CATEGORY="single"
SEED_NUMBER=42
BATCH_SIZES=(64)
EPOCHS=75
OBJECTIVE="auroc"
METRICS=("auroc" "auprc" "f1")

# Uncomment below to switch to Mortality
#WANDB_PROJECT="mimic_pheno_finetuning"
#TASK="phenotyping"
WANDB_PROJECT="mimic_mortality_finetuning"
TASK="mortality"

STATE_DICT="/users/awang463/data/awang463/missing_modalities/models/contrastive/0_masked_global_1e-4-wd_0.005-gamma_2.0_dropped_filler/masked_global_filler_prob_0.0_lr_0.0001-wd_0.005-temp_0.07_20251209-213342_final.pth"
WEIGH_LOSS=false

# --- DYNAMIC CONFIGURATION BASED ON INIT_MODE ---
if [ "$INIT_MODE" = "baseline" ]; then
    MODEL_NAME="baseline"
    FREEZE=false
else
    MODEL_NAME="ovo"
    FREEZE=true  # Freezes the contrastive encoder for linear probing
fi

WANDB_NAME="finetune_${TASK}_${MODEL_NAME}_${FUSION_METHOD}_lr${LEARNING_RATE}"
SAVE_PREFIX="/users/awang463/scratch/finetune_${TASK}_${MODEL_NAME}_${FUSION_METHOD}_lr${LEARNING_RATE}"

echo "===================================================="
echo "SLURM JOB ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "RUNNING FINETUNE JOB WITH PARAMS:"
echo "  - Task: $TASK"
echo "  - Init Mode (Model): $MODEL_NAME"
echo "  - Fusion Method: $FUSION_METHOD"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Freeze Encoder: $FREEZE"
echo "===================================================="

# --- CONSTRUCT COMMAND ---
COMMAND="python3 model/finetune.py \
    --model_name $MODEL_NAME \
    --modalities ${MODALITIES[@]} \
    --learning_rate $LEARNING_RATE \
    --seed_category $SEED_CATEGORY \
    --fusion_method $FUSION_METHOD \
    --batch_size ${BATCH_SIZES[@]} \
    --epochs $EPOCHS \
    --objective $OBJECTIVE \
    --state_dict $STATE_DICT \
    --metrics ${METRICS[@]} \
    --wandb_project $WANDB_PROJECT \
    --task $TASK \
    --save_prefix $SAVE_PREFIX \
    --wandb_name $WANDB_NAME"

# Add contrastive weights if applicable
if [ "$INIT_MODE" = "ovo" ]; then
    COMMAND+=" --state_dict $STATE_DICT"
fi

if [ "$FREEZE" = true ]; then
    COMMAND+=" --freeze"
fi

if [ "$WEIGH_LOSS" = true ]; then
    COMMAND+=" --weigh_loss"
fi

if [ "$SEED_CATEGORY" = "single" ]; then
    COMMAND+=" --seed_number $SEED_NUMBER"
fi

# Run the command
echo "Running command: $COMMAND"
$COMMAND