#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 20480 // Size of the matrices (N x N)

void matrixMul(float *A, float *B, float *C, int n) {
    float *d_A, *d_B, *d_C;
    cublasHandle_t handle;

    // Allocate device memory
    cudaMalloc((void **)&d_A, n * n * sizeof(float));
    cudaMalloc((void **)&d_B, n * n * sizeof(float));
    cudaMalloc((void **)&d_C, n * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasCreate(&handle);

    // Start timing
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call cuBLAS function to perform the matrix multiplication
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, 
                &alpha, 
                d_A, n, 
                d_B, n, 
                &beta, 
                d_C, n);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)(rand() % 10);
            B[i * N + j] = (float)(rand() % 10);
            C[i * N + j] = 0; // Initialize result matrix
        }
    }

    // Perform matrix multiplication
    matrixMul(A, B, C, N);

    // Print a small part of the result for verification (optional)
    printf("Result matrix C (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", C[i]);
    }
    printf("\n");

    // Clean up
    free(A);
    free(B);
    free(C);
    return 0;
}

