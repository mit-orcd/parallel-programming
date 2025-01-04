#include "mpi.h"

main(int argc, char** argv){

  int my_rank, i;
  //int n=10;
  int n=5000;
  int n_send1[n], n_send2[n], n_recv1[n], n_recv2[n];
  //int  numbertosend=36;
  //printf("The Number is: %d\n", numbertoreceive);

  MPI_Init(&argc, &argv);
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank==0){
    for(i=0; i<n; i++){
      n_send1[i]=i;
      n_recv2[i]=0;
    }
    MPI_Send( &n_send1, n, MPI_INT, 1, 10, MPI_COMM_WORLD);
    MPI_Recv( &n_recv2, n, MPI_INT, 1, 11, MPI_COMM_WORLD, &status);
    //printf("Number received on worker %d is: %d\n", my_rank, n_recv2);
  }
  else if (my_rank==1){
    for(i=0; i<n; i++){
      n_send2[i]=i+100;
      n_recv1[i]=0;
    }
    MPI_Recv( &n_recv1, n, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    MPI_Send( &n_send2, 1, MPI_INT, 0, 11, MPI_COMM_WORLD);
    //printf("Number received on worker %d is: %d\n", my_rank, n_recv1);
  }

  printf("Job done on rank %d. \n", my_rank);
 
  MPI_Finalize(); 


}
