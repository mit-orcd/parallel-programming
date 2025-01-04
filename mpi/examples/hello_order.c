//#include <stdio.h>
#include "mpi.h"

main(int argc, char** argv){

  int my_rank, my_size, i;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &my_size);

  for(i=0; i<my_size; i++){
     if(i==my_rank) printf("Hello from %d.\n", my_rank);
     MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();

}
