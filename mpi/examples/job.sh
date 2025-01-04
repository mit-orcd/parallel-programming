#!/bin/bash
#SBATCH -t 120 
#SBATCH -p mit_normal 
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=20GB 
#SBATCH -o output/%N-%J.out

module load openmpi/4.1.4

#mpirun -np 2 send_recv 
mpirun -np 2 unlock
#mpirun -np 2 small_send
#mpirun -np 2 deadlock_send
#mpirun -np 2 deadlock_recv
#mpirun -np 4 hello_order
