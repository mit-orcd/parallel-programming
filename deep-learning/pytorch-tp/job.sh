#!/bin/bash
#SBATCH -p mit_normal_gpu   #  mit_normal_gpu ou_bcs_low ou_sloan_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=30GB
#SBATCH --gres=gpu:4
#SBATCH -t 60
#SBATCH -o output/out.4gpu-%N-%J

# set up environment
module load gcc/12.2.0
module load cuda/12.4.0
module load miniforge/23.11.0-0
source activate ds

echo "=== start ==="
echo "Launching ${1:-fsdp_tp_example.py} with ${2:-4} gpus"
torchrun --nnodes=1 --nproc_per_node=${2:-4} --rdzv_id=101 --rdzv_endpoint="localhost:5972" ${1:-fsdp_tp_example.py}

