#include <mpi.h>
#include <math.h>

int main( int argc, char **argv ){

  int rank, numprocs;
  int evensum, oddsum; 

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs );
  MPI_Comm_rank(MPI_COMM_WORLD, &rank );

  if (rank % 2 == 0) // Even processes
  {
    MPI_Allreduce(&rank, &evensum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0){ 
       printf("evensum = %d\n", evensum);
       //printf("oddsum = %d\n", oddsum);
    }
  }
  else // Odd processes
  {
    MPI_Allreduce(&rank, &oddsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 1){ 
       printf("oddsum = %d\n", oddsum);
       //printf("evensum = %d\n", evensum);
    }
  }

  MPI_Finalize();

}
