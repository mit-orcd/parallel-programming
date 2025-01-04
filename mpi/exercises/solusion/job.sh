#!/bin/bash
#SBATCH -t 120 
#SBATCH -p mit_normal 
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=20GB 

module load openmpi/4.1.4

mpirun -np 4 hostname
time mpirun -np 4 ./calc_pi_mpi < input
