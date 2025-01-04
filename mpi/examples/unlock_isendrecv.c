#include "mpi.h"

main(int argc, char** argv){

  int my_rank, i;
  int n=5000;
  int n_send1[n], n_send2[n], n_recv1[n], n_recv2[n];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Request send_request, recv_request;
  MPI_Status status;

  if (my_rank==0){
    for(i=0; i<n; i++){
      n_send1[i]=i;
      n_recv2[i]=0;
    }
    //MPI_Irecv( &n_recv2, n, MPI_INT, 1, 11, MPI_COMM_WORLD, &recv_request);
    MPI_Isend( &n_send1, n, MPI_INT, 1, 10, MPI_COMM_WORLD, &send_request);
    MPI_Irecv( &n_recv2, n, MPI_INT, 1, 11, MPI_COMM_WORLD, &recv_request);
  }
  else if (my_rank==1){
    for(i=0; i<n; i++){
      n_send2[i]=i*10;
      n_recv1[i]=0;
    }
    //MPI_Irecv( &n_recv1, n, MPI_INT, 0, 10, MPI_COMM_WORLD, &recv_request);
    MPI_Isend( &n_send2, n, MPI_INT, 0, 11, MPI_COMM_WORLD, &send_request);
    MPI_Irecv( &n_recv1, n, MPI_INT, 0, 10, MPI_COMM_WORLD, &recv_request);
  }

  MPI_Wait(&send_request, &status);

  n_send1[n-1]=201;
  n_send2[n-1]=301;

  MPI_Wait(&recv_request, &status);

  //printf("Job done on rank %d. \n", my_rank);
  if(my_rank==0) printf("The last element on rank %d is %d. \n", my_rank, n_recv2[n-1]);
  if(my_rank==1) printf("The last element on rank %d is %d. \n", my_rank, n_recv1[n-1]);
 
  MPI_Finalize(); 


}
