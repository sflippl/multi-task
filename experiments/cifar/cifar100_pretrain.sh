#!/bin/sh
#SBATCH --job-name=cifar1
#SBATCH --partition=gpu_saxe
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-49
#SBATCH --output=slurm/slurm-%A_%a.out

mkdir -p /nfs/nhome/live/cdomine/old_sam/multi-task/slurm
cd /nfs/nhome/live/cdomine/old_sam/multi-task/
source ~/.bashrc
conda activate NPG-env

python experiments/cifar/cifar100_pretrain.py $SLURM_ARRAY_TASK_ID
