#!/bin/sh
#SBATCH --job-name=cifar3
#SBATCH --partition=gpu_saxe
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --array=0-1900
#SBATCH --output=slurm/slurm-%A_%a.out
# Create the directory immediately before running
mkdir -p /nfs/nhome/live/cdomine/old_sam/multi-task/slurm
cd /nfs/nhome/live/cdomine/old_sam/multi-task/
source ~/.bashrc
conda activate NPG-env

python experiments/cifar/cifar100_main_2.py $((SLURM_ARRAY_TASK_ID+1000))
