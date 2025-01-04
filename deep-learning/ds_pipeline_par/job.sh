#!/bin/bash
#SBATCH -p ou_sloan_gpu   #  mit_normal_gpu # ou_bcs_low
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

# run with 2 pipeline stages (-p)
deepspeed train.py --deepspeed_config=ds_config.json -p 2 --steps=200

