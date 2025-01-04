#!/bin/bash
#SBATCH -p mit_normal_gpu #ou_bcs_low  # mit_normal_gpu
#SBATCH --job-name=ddp
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:h100:2
#SBATCH -o output/%N-%J.out

module load miniforge/23.11.0-0
source activate ds

#usage: multigpu.py [-h] [--batch_size BATCH_SIZE] total_epochs save_every
echo "======== Run on one GPU ======"
python single_gpu.py --batch_size=1024 100 20
echo "======== Run on one multiple GPUs ======"
python multigpu.py --batch_size=1024 100 20

