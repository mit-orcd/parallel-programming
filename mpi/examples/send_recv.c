#include "mpi.h"

main(int argc, char** argv){

  int my_rank, numbertoreceive, numbertosend;

  MPI_Init(&argc, &argv);
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank==0){
    numbertosend=36;
    MPI_Send( &numbertosend, 1, MPI_INT, 1, 10, MPI_COMM_WORLD);
  }
  else if (my_rank==1){
    MPI_Recv( &numbertoreceive, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    printf("Number received is: %d\n", numbertoreceive);
  }
 
  MPI_Finalize(); 

}
