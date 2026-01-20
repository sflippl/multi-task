#!/bin/sh
#SBATCH --job-name=cifar3
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --array=0-2999%15
#SBATCH --output=fc_res_slurm/slurm-%A_%a.out

mkdir -p /nfs/nhome/live/cdomine/old_sam/multi-task/slurm
cd /nfs/nhome/live/cdomine/old_sam/multi-task/
source ~/.bashrc
ls$conda activate NPG-env

python experiments/cifar/cifar100_main_2.py $SLURM_ARRAY_TASK_ID
