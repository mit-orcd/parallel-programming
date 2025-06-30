#!/bin/bash
#SBATCH -p mit_preemptable  # mit_normal_gpu 
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=30GB
#SBATCH --gres=gpu:4  #h100:4  # 4
#SBATCH -t 60
#SBATCH -o output/out.4gpu-%N-%J

# set up environment
module load miniforge/23.11.0-0
source activate ds

echo "=== start ==="
echo $SLURM_NNODES
echo $SLURM_NTASKS
echo $SLURM_JOB_ID

torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_endpoint="localhost:1234" fsdp_tp_example.py  

