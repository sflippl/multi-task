#!/bin/sh
#SBATCH --job-name=cifar1
#SBATCH --partition=gpu_saxe
#SBATCH --mem=20G
#SBATCH --array=0-600%4
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=slurm_train_2_saxe/slurm-%A_%a.out
mkdir -p /nfs/nhome/live/cdomine/old_sam/multi-task/slurm
cd /nfs/nhome/live/cdomine/old_sam/multi-task/
source ~/.bashrc
conda activate NPG-env
python experiments/cifar/cifar100_pretrain.py $SLURM_ARRAY_TASK_ID