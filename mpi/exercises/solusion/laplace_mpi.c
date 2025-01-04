/****************************************************************
 * Laplace MPI C Version                                         
 *                                                               
 * T is initially 0.0                                            
 * Boundaries are as follows                                     
 *                                                               
 *                T                      4 sub-grids            
 *   0  +-------------------+  0    +-------------------+       
 *      |                   |       |                   |           
 *      |                   |       |-------------------|         
 *      |                   |       |                   |      
 *   T  |                   |  T    |-------------------|             
 *      |                   |       |                   |     
 *      |                   |       |-------------------|            
 *      |                   |       |                   |   
 *   0  +-------------------+ 100   +-------------------+         
 *      0         T       100                                    
 *                                                                 
 *                                                                 
 *******************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define COLUMNS      1000
#define ROWS_GLOBAL  1000        // this is a "global" row count
#define NPROC            4        // number of processors
#define ROWS (ROWS_GLOBAL/NPROC)  // number of real local rows

// communication tags
#define DOWN     100
#define UP       101   

#define tolerance 0.01

double A_new[ROWS+2][COLUMNS+2];
double A[ROWS+2][COLUMNS+2];

void initialize(int nproc, int my_rank);
void track_progress(int iter);


int main(int argc, char *argv[]) {

    int i, j;
    int max_iterations;
    int iteration=1;
    double dA;
    struct timeval start_time, stop_time, elapsed_time;

    int        nproc;                // number of PEs
    int        my_rank;           // my PE number
    double     dA_global=100;       // delta t across all PEs
    MPI_Status status;              // status returned by MPI calls

    // the usual MPI startup routines
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // verify only NPROC PEs are being used
    if(nproc != NPROC) {
      if(my_rank==0) {
        printf("This code must be run with %d PEs\n", NPROC);
      }
      MPI_Finalize();
      exit(1);
    }

    // PE 0 asks for input
    if(my_rank==0) {
      printf("Maximum iterations [100-4000]?\n");
      fflush(stdout); // Not always necessary, but can be helpful
      scanf("%d", &max_iterations);
    }

    // bcast max iterations to other PEs
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank==0) gettimeofday(&start_time,NULL);

    initialize(nproc, my_rank);

    while ( dA_global > tolerance && iteration <= max_iterations ) {

        // main calculation: average my four neighbors
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                A_new[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] +
                                            A[i][j+1] + A[i][j-1]);
            }
        }

        // COMMUNICATION PHASE: send ghost rows for next iteration

        // send bottom real row down
        if(my_rank != nproc-1){             //unless we are bottom PE
            MPI_Send(&A_new[ROWS][1], COLUMNS, MPI_DOUBLE, my_rank+1, DOWN, MPI_COMM_WORLD);
        }

        // receive the bottom row from above into our top ghost row
        if(my_rank != 0){                  //unless we are top PE
            MPI_Recv(&A[0][1], COLUMNS, MPI_DOUBLE, my_rank-1, DOWN, MPI_COMM_WORLD, &status);
        }

        // send top real row up
        if(my_rank != 0){                    //unless we are top PE
            MPI_Send(&A_new[1][1], COLUMNS, MPI_DOUBLE, my_rank-1, UP, MPI_COMM_WORLD);
        }

        // receive the top row from below into our bottom ghost row
        if(my_rank != nproc-1){             //unless we are bottom PE
            MPI_Recv(&A[ROWS+1][1], COLUMNS, MPI_DOUBLE, my_rank+1, UP, MPI_COMM_WORLD, &status);
        }

        dA = 0.0;

        for(i = 1; i <= ROWS; i++){
            for(j = 1; j <= COLUMNS; j++){
	        dA = fmax( fabs(A_new[i][j]-A[i][j]), dA);
	        A[i][j] = A_new[i][j];
            }
        }

        // find global dA                                                        
        MPI_Reduce(&dA, &dA_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dA_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // periodically print test values - only for PE in lower corner
        if((iteration % 100) == 0) {
            if (my_rank == nproc-1){
                track_progress(iteration);
	    }
        }

	iteration++;
    }

    // Slightly more accurate timing and cleaner output 
    MPI_Barrier(MPI_COMM_WORLD);

    // PE 0 finish timing and output values
    if (my_rank==0){
        gettimeofday(&stop_time,NULL);
	timersub(&stop_time, &start_time, &elapsed_time);

	printf("\nMax error at iteration %d was %f\n", iteration-1, dA_global);
	printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    }

    MPI_Finalize();
}



void initialize(int nproc, int my_rank){

    double tMin, tMax;  //Local boundary limits
    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            A[i][j] = 0.0;
        }
    }

    // Local boundry condition endpoints
    tMin = (my_rank)*100.0/nproc;
    tMax = (my_rank+1)*100.0/nproc;

    // Left and right boundaries
    for (i = 0; i <= ROWS+1; i++) {
      A[i][0] = 0.0;
      A[i][COLUMNS+1] = tMin + ((tMax-tMin)/ROWS)*i;
    }

    // Top boundary (PE 0 only)
    if (my_rank == 0)
      for (j = 0; j <= COLUMNS+1; j++)
	A[0][j] = 0.0;

    // Bottom boundary (Last PE only)
    if (my_rank == nproc-1)
      for (j=0; j<=COLUMNS+1; j++)
	A[ROWS+1][j] = (100.0/COLUMNS) * j;

}


// only called by last PE
void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);

    // output global coordinates so user doesn't have to understand decomposition
    for(i = 5; i >= 0; i--) {
      printf("[%d,%d]: %5.2f  ", ROWS_GLOBAL-i, COLUMNS-i, A_new[ROWS-i][COLUMNS-i]);
    }
    printf("\n");
}
