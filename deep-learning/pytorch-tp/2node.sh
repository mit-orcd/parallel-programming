#!/bin/bash
#SBATCH -p mit_normal_gpu 
#SBATCH -q unlimited
#SBATCH --job-name=tp-2nodes
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4   # 4 , h200:4
#SBATCH --mem=20GB
#SBATCH -o output/%x-%N-%J.out

# Load conda env. Both work.
module load miniforge/23.11.0-0
source activate ds

echo "======== Test PyTorch with CUDA ======"
which python
which torchrun
python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.nccl.version())'

echo "======== Print host names ======"
srun hostname
srun hostname --ip-address

# Get IP address
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo $nodes
nodes_array=($nodes)
master_node=${nodes_array[0]}
master_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)
echo $master_node_ip

echo "======== Run on multiple GPUs ======"
echo "WORLD_SIZE = $WORLD_SIZE"

# use srun to lauch torchrun on two nodes
srun torchrun --nnodes=$SLURM_NNODES \
    --nproc-per-node=$SLURM_CPUS_PER_TASK \
    --rdzv-id=$SLURM_JOB_ID   \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$master_node_ip:12345 \
    fsdp_tp_example.py

