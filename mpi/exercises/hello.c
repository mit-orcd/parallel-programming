//#include <stdio.h>
#include "mpi.h"

main(int argc, char** argv){

  int my_rank, my_size;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &my_size);
  printf("Hello from %d of %d.\n", my_rank, my_size);

  MPI_Finalize();

}
