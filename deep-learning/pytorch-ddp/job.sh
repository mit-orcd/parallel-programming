#!/bin/bash
#SBATCH -p mit_normal_gpu # mit_preemptable # mit_normal_gpu 
#SBATCH --job-name=ddp
#SBATCH -N 1
#SBATCH -t 60
#SBATCH -n 2     # should be the same as number of GPUs
#SBATCH --mem=20GB
#SBATCH --gres=gpu:2    # h100:4 , 4
#SBATCH -o output/%N-%J.out

module load miniforge/23.11.0-0
source activate ds

# Usage: multigpu.py [-h] [--batch_size BATCH_SIZE] total_epochs save_every
echo "======== Run on multiple GPUs ======"
python multigpu.py --batch_size=1024 100 20
echo "======== Run on multiple GPUs with torchrun ======"
torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_NTASKS --rdzv_id=$SLURM_JOB_ID --rdzv_endpoint="localhost:1230" multigpu_torchrun.py --batch_size=1024 100 20

