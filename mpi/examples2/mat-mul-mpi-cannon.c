#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4 // Size of the matrices (N x N)

// Function to print a matrix
void print_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int A[N][N], B[N][N], C[N][N] = {0};
    int local_A[N][N], local_B[N][N], local_C[N][N] = {0};
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assuming size is a perfect square
    int sqrt_p = (int)sqrt(size);
    
    // Initialize matrices only in rank 0
    if (rank == 0) {
        // Initialize matrix A
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j; // Example initialization
            }
        }
        // Initialize matrix B
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B[i][j] = i * j; // Example initialization
            }
        }
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter matrix A to all processes
    MPI_Scatter(A, N*N/sqrt_p/sqrt_p, MPI_INT, local_A, N*N/sqrt_p/sqrt_p, MPI_INT, 0, MPI_COMM_WORLD);

    // Initial alignment of local_A and local_B
    int row_shift = rank / sqrt_p;
    int col_shift = rank % sqrt_p;

    // Shift local_A
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Sendrecvrecv(local_A, N/sqrt_p*N/sqrt_p, MPI_INT, (rank - row_shift + sqrt_p) % sqrt_p + col_shift * sqrt_p, 0,
                     local_A, N/sqrt_p*N/sqrt_p, MPI_INT, (rank + row_shift) % sqrt_p + col_shift * sqrt_p, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Shift local_B
    MPI_Sendrecvrecv(local_B, N/sqrt_p*N/sqrt_p, MPI_INT, (rank - col_shift + sqrt_p) % sqrt_p, 0,
                     local_B, N/sqrt_p*N/sqrt_p, MPI_INT, (rank + col_shift) % sqrt_p, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Cannon's algorithm main computation
    for (int step = 0; step < sqrt_p; step++) {
        // Multiply local_A and local_B, accumulating the result in local_C
        for (int i = 0; i < N/sqrt_p; i++) {
            for (int j = 0; j < N/sqrt_p; j++) {
                for (int k = 0; k < N/sqrt_p; k++) {
                    local_C[i][j] += local_A[i][k] * local_B[k][j];
                }
            }
        }

        // Shift local_A to the left
        MPI_Sendrecv(local_A, N/sqrt_p*N/sqrt_p, MPI_INT,
                     (rank - row_shift + sqrt_p) % sqrt_p + col_shift * sqrt_p, 0,
                     local_A, N/sqrt_p*N/sqrt_p, MPI_INT,
                     (rank + row_shift) % sqrt_p + col_shift * sqrt_p, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Shift local_B upwards
        MPI_Sendrecv(local_B, N/sqrt_p*N/sqrt_p, MPI_INT,
                     (rank - col_shift + sqrt_p) % sqrt_p, 0,
                     local_B, N/sqrt_p*N/sqrt_p, MPI_INT,
                     (rank + col_shift) % sqrt_p, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Gather results into matrix C in rank 0
    MPI_Gather(local_C, N*N/sqrt_p/sqrt_p, MPI_INT, C, N*N/sqrt_p/sqrt_p, MPI_INT, 0, MPI_COMM_WORLD);

    // Print result in rank 0
    if (rank == 0) {
        printf("Result matrix C:\n");
        print_matrix(C);
    }

    MPI_Finalize();
    return 0;
}

