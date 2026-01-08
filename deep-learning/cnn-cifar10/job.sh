#!/bin/bash
#SBATCH -p mit_normal_gpu   # ou_bcs_low 
#SBATCH --gres=gpu:1   # h100, l40s
#SBATCH -t 30
#SBATCH -n 2
#SBATCH --mem=30GB
#SBATCH -o output/%N-%J.out

# set env
module load cuda/12.4.0
module load miniforge/23.11.0-0
source activate ds
#source activate torch-gpu

echo "~~~~~~~~ Run the pytorch code on CPU ~~~~~~~~~"
time python cnn_cifar10_cpu.py 
echo "~~~~~~~~ Run the pytorch code on GPU ~~~~~~~~~"
time python cnn_cifar10_gpu.py 

