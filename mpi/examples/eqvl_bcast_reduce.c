#include <stdio.h>
#include "mpi.h"

main(int argc, char** argv){

  int my_rank, numbertoreceive, numbertosend=2,index, result=0;

  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank==0)
    for (index=1; index<4; index++)
    MPI_Send( &numbertosend, 1,MPI_INT, index, 10,MPI_COMM_WORLD);
  else{
    MPI_Recv( &numbertoreceive, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    result = numbertoreceive * my_rank;
  }

  for (index=1; index<4; index++){
    MPI_Barrier(MPI_COMM_WORLD);
    if (index==my_rank) printf("PE %d's result is %d.\n", my_rank, result);
  }

  if (my_rank==0){
    for (index=1; index<4; index++){
      MPI_Recv( &numbertoreceive, 1,MPI_INT,index,10, MPI_COMM_WORLD, &status);
      result += numbertoreceive;
    }
    printf("Total is %d.\n", result);
  }
  else
    MPI_Send( &result, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);

  MPI_Finalize();

}
