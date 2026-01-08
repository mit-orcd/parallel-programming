#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000000  // Size of the vectors

// Kernel function to perform SAXPY (Y = alpha * X + Y)
__global__ void saxpy_kernel(int n, float alpha, float *X, float *Y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index

    // Ensure the thread index is within bounds
    if (i < n) {
        Y[i] = alpha * X[i] + Y[i];
    }
}

int main() {
    int n = N;  // Size of the vectors
    float alpha = 2.0f;  // Scalar value

    // Allocate memory for the vectors on the host (CPU)
    float *h_X = (float *)malloc(n * sizeof(float));
    float *h_Y = (float *)malloc(n * sizeof(float));

    // Initialize the host vectors
    for (int i = 0; i < n; i++) {
        h_X[i] = (float)(i + 1);  // X = [1, 2, 3, ..., N]
        h_Y[i] = (float)(n - i);  // Y = [N, N-1, ..., 1]
    }

    // Allocate memory for the vectors on the device (GPU)
    float *d_X, *d_Y;
    cudaMalloc((void **)&d_X, n * sizeof(float));
    cudaMalloc((void **)&d_Y, n * sizeof(float));

    // Copy the input vectors from host to device
    cudaMemcpy(d_X, h_X, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the execution configuration (1D grid and block)
    int blockSize = 256;  // Number of threads per block
    //int numBlocks = (n + blockSize - 1) / blockSize;  // Number of blocks
    int numBlocks = n / blockSize;  // Number of blocks

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start time
    cudaEventRecord(start);

    // Launch the SAXPY kernel
    saxpy_kernel<<<numBlocks, blockSize>>>(n, alpha, d_X, d_Y);

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Record the stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result vector Y from device to host
    cudaMemcpy(h_Y, d_Y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Optionally, print the first few elements of the result vector
    printf("Resulting Y vector after SAXPY operation (first 10 elements):\n");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("Y[%d] = %f\n", i, h_Y[i]);
    }

    // Output the time taken by the kernel
    printf("\nTime taken for SAXPY operation: %f ms\n", milliseconds);

    // Clean up memory
    free(h_X);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_Y);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

