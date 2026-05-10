#!/bin/bash
#SBATCH --job-name=mnar_horseshoe
#SBATCH --array=2-5                 
#SBATCH --time=4-00:00:00              
#SBATCH --mem=32G               
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stan_chain_%a.out
#SBATCH --partition=batch

# Pass the array ID to the R script
Rscript bayesian_inference.R $SLURM_ARRAY_TASK_ID
