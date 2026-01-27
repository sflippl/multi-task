#!/bin/bash
#SBATCH --job-name=cifar1
#SBATCH --partition=gpu100
#SBATCH --mem=20G
#SBATCH --array=0-14000%7
#SBATCH --gres=gpu:1
#SBATCH --reservation=locatgrp_211
#SBATCH --time=10:00:00
#SBATCH --output=slurm_block_123_alpha/slurm-%A_%a.out


# 1. Clean environment and move to project
cd /nfs/scistore19/locatgrp/cdomine/multi-task/

# 2. Source the Miniforge profile (Crucial for SLURM)
# This defines the 'conda' command for this specific shell session
export CONDA_PATH="/nfs/scistore19/locatgrp/cdomine/miniforge3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"

# 3. Activate by name
conda activate NPG-env

# 4. Verification (Check logs to see if this points to your miniforge folder)
which python
python -c "import torch; print('Torch version:', torch.__version__)"

python experiments/cifar/cifar100_main_2.py $SLURM_ARRAY_TASK_ID
