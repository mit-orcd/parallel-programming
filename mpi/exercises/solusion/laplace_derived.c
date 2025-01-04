/****************************************************************
 * Laplace MPI C Version                                         
 *                                                               
 * use derived datatypes
 *
 *******************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define ROWS_GLOBAL  1000        // this is a "global" row count
#define COLUMNS_GLOBAL   1000
#define NPROC          4        // number of processors
#define ROWS (ROWS_GLOBAL/2)  // number of real local rows
#define COLUMNS (COLUMNS_GLOBAL/2)  // number of real local rows

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

    // define datatypes, will be used for communication
    MPI_Datatype sub_row, sub_column ; 
    MPI_Type_contiguous( COLUMNS, MPI_DOUBLE, &sub_row );
    MPI_Type_vector ( ROWS, 1, COLUMNS+2, MPI_DOUBLE , &sub_column ); 
    MPI_Type_commit ( &sub_row ); 
    MPI_Type_commit ( &sub_column );

    while ( dA_global > tolerance && iteration <= max_iterations ) {

        // main calculation: average my four neighbors
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                A_new[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] +
                                            A[i][j+1] + A[i][j-1]);
            }
        }

        // COMMUNICATION PHASE: send ghost rows and columns for next iteration
        if(my_rank==0){
            MPI_Send(&A_new[1][COLUMNS], 1, sub_column, 1, 801, MPI_COMM_WORLD);
            MPI_Recv(&A[1][COLUMNS+1], 1, sub_column, 1, 810, MPI_COMM_WORLD, &status);

            MPI_Send(&A_new[ROWS][1], 1, sub_row, 2, 802, MPI_COMM_WORLD);
            MPI_Recv(&A[ROWS+1][1], 1, sub_row, 2, 820, MPI_COMM_WORLD, &status);
        }          
 
        if(my_rank==1){
            MPI_Recv(&A[1][0], 1, sub_column, 0, 801, MPI_COMM_WORLD, &status);
            MPI_Send(&A_new[1][1], 1, sub_column, 0, 810, MPI_COMM_WORLD);
            
            MPI_Send(&A_new[ROWS][1], 1, sub_row, 3, 813, MPI_COMM_WORLD);
            MPI_Recv(&A[ROWS+1][1], 1, sub_row, 3, 831, MPI_COMM_WORLD, &status);
        }           

        if(my_rank==2){
            MPI_Recv(&A[0][1], 1, sub_row, 0, 802, MPI_COMM_WORLD, &status);
            MPI_Send(&A_new[1][1], 1, sub_row, 0, 820, MPI_COMM_WORLD);

            MPI_Send(&A_new[1][COLUMNS], 1, sub_column, 3, 823, MPI_COMM_WORLD);
            MPI_Recv(&A[1][COLUMNS+1], 1, sub_column, 3, 832, MPI_COMM_WORLD, &status);
        }           

        if(my_rank==3){
            MPI_Recv(&A[0][1], 1, sub_row, 1, 813, MPI_COMM_WORLD, &status);
            MPI_Send(&A_new[1][1], 1, sub_row, 1, 831, MPI_COMM_WORLD);

            MPI_Recv(&A[1][0], 1, sub_column, 2, 823, MPI_COMM_WORLD, &status);
            MPI_Send(&A_new[1][1], 1, sub_column, 2, 832, MPI_COMM_WORLD);
        }           
 /*       // send bottom real row down
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
*/

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

        // periodically print test values - only for PE in low-right corner
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

    // all initial values are zero
    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            A[i][j] = 0.0;
        }
    }

    // boundary condition
    if (my_rank == 1)
      for (i = 0; i <= ROWS+1; i++)  A[i][COLUMNS+1] = 50./ROWS*i;

    if (my_rank == 2)
      for (j = 0; j <= COLUMNS+1; j++)  A[ROWS+1][j] = 50./COLUMNS*j;

    if (my_rank == 3) {
      for (i = 0; i <= ROWS+1; i++)  A[i][COLUMNS+1] = 50.+50./ROWS*i;
      for (j = 0; j <= COLUMNS+1; j++)  A[ROWS+1][j] = 50.+50./COLUMNS*j;
    }

}


// only called by last PE
void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);

    // output global coordinates so user doesn't have to understand decomposition
    for(i = 5; i >= 0; i--) {
      printf("[%d,%d]: %5.2f  ", ROWS_GLOBAL-i, COLUMNS_GLOBAL-i, A_new[ROWS-i][COLUMNS-i]);
    }
    printf("\n");
}
