#!/bin/sh
#SBATCH --job-name=ensd1
#SBATCH --partition=cpu
#SBATCH --mem=20G
#SBATCH --time=16:00:00
#SBATCH --array=0-3000
#SBATCH --output=slurms_ensd_sam/slurm-%A_%a.out

python experiments/cifar/cifar100_ensd_sam.py $SLURM_ARRAY_TASK_ID
