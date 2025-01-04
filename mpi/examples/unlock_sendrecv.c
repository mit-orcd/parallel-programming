#include "mpi.h"

main(int argc, char** argv){

  int my_rank, i;
  int n=5000;
  int n_send1[n], n_send2[n], n_recv1[n], n_recv2[n];

  MPI_Init(&argc, &argv);
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank==0){
    for(i=0; i<n; i++){
      n_send1[i]=i;
      n_recv2[i]=0;
    }
    MPI_Sendrecv( &n_send1, n, MPI_INT, 1, 10, &n_recv2, n, MPI_INT, 1, 11, MPI_COMM_WORLD, &status);
  }
  else if (my_rank==1){
    for(i=0; i<n; i++){
      n_send2[i]=i+100;
      n_recv1[i]=0;
    }
    MPI_Sendrecv( &n_send2, n, MPI_INT, 0, 11, &n_recv1, n, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
  }

  printf("Job done on rank %d. \n", my_rank);
 
  MPI_Finalize(); 


}
