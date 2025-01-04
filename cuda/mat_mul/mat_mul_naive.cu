#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 20480 // Size of the matrices (N x N)

__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

void matrixMul(float *A, float *B, float *C, int n) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void **)&d_A, n * n * sizeof(float));
    cudaMalloc((void **)&d_B, n * n * sizeof(float));
    cudaMalloc((void **)&d_C, n * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the kernel
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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

