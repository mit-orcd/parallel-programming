#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4 // Size of the matrices

void print_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int A[N][N], B[N][N], C[N][N];
    int local_A[N][N], local_C[N][N];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N) {
        if (rank == 0) {
            printf("Please run with %d processes.\n", N);
        }
        MPI_Finalize();
        return 0;
    }

    // Initialize matrices A and B
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j;
                B[i][j] = i - j;
            }
        }
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of matrix A to all processes
    MPI_Scatter(A, N*N/size, MPI_INT, local_A, N*N/size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local computation
    for (int i = 0; i < N/size; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    // Gather the local matrices C from all processes
    MPI_Gather(local_C, N*N/size, MPI_INT, C, N*N/size, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the result matrix C
    if (rank == 0) {
        printf("Matrix A:\n");
        print_matrix(A);
        printf("Matrix B:\n");
        print_matrix(B);
        printf("Matrix C (Result):\n");
        print_matrix(C);
    }

    MPI_Finalize();
    return 0;
}

