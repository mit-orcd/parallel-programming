#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv){

  int rank, nproc, i;
  int m, n=100;
  double sub_avg=0., global_avg=0.;
  double * array = NULL, * sub_avgs = NULL;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  m = (int) n/nproc;  // chunk size

  //double * array = NULL;
  //double * sub_avgs = NULL;

  // initial data on worker 0
  if(rank==0){
     array = (double *) malloc(n*sizeof(double));
     for(i=0; i<n; i++){
       array[i]=(double)i;
     }
  }


  double * chunk = (double *) malloc(m*sizeof(double));
  MPI_Scatter(array, m, MPI_DOUBLE, chunk, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  // calculate sub averages on all workers
  for(i=0; i<m; i++){
     sub_avg += chunk[i]; 
     //printf("chunk[i] on %d is: %f\n", rank, chunk[i]);
  }
  sub_avg = sub_avg/(double)m;
  printf("The sub average on %d is: %f\n", rank, sub_avg);

  MPI_Barrier(MPI_COMM_WORLD);

  if(rank==0) sub_avgs = (double *) malloc(nproc*sizeof(double));
  MPI_Gather(&sub_avg, 1, MPI_DOUBLE, sub_avgs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  


  // calculate global average on worker 0
  if(rank==0){
     for(i=0; i<nproc; i++){
        global_avg += sub_avgs[i];
     }
     printf("The global average is: %f\n", global_avg/(double)nproc);
  }

  free(chunk);
  if(rank==0){ 
     free(array);
     free(sub_avgs);
  }

  MPI_Finalize();
  return 0;

}
