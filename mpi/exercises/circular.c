//#include <stdio.h>
#include "mpi.h"

main(int argc, char** argv){

  int my_rank, my_size;
  int n_send, n_recv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &my_size);

  n_send = my_rank;
  //printf("Process %d 's n_send is %d\n", my_rank, n_send);
  

  MPI_Send(&n_send, 1, MPI_INT, (my_rank + 1) % my_size, 0, MPI_COMM_WORLD);

  if (my_rank != 0) {
    MPI_Recv(&n_recv, 1, MPI_INT, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process %d received number %d from process %d\n", my_rank, n_recv, my_rank - 1);
  } 
  else { 
    MPI_Recv(&n_recv, 1, MPI_INT, my_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process %d received number %d from process %d\n", my_rank, n_recv, my_size - 1);
  }

  //MPI_Send(&n_send, 1, MPI_INT, (my_rank + 1) % my_size, 0, MPI_COMM_WORLD);  // deadlock
  
    // Now process 0 can receive from the last process.
//  if (my_rank == 0) { 
//    MPI_Recv(&n_recv, 1, MPI_INT, my_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    printf("Process %d received number %d from process %d\n", my_rank, n_recv, my_size - 1);
//  }


  MPI_Finalize();

}
