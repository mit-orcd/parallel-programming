#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000000  // Size of the vectors

// Function to perform SAXPY (Y = alpha * X + Y) in serial
void saxpy(int n, float alpha, float *X, float *Y) {
    for (int i = 0; i < n; i++) {
        Y[i] = alpha * X[i] + Y[i];
    }
}

int main() {
    int n = N;  // Size of the vectors
    float alpha = 2.0f;  // Scalar value

    // Allocate memory for the vectors
    float *X = (float *)malloc(n * sizeof(float));
    float *Y = (float *)malloc(n * sizeof(float));

    // Initialize the vectors X and Y
    for (int i = 0; i < n; i++) {
        X[i] = (float)(i + 1);  // X = [1, 2, 3, ..., N]
        Y[i] = (float)(n - i);  // Y = [N, N-1, ..., 1]
    }

    // Measure the start time
    clock_t start = clock();

    // Perform the SAXPY operation in serial
    saxpy(n, alpha, X, Y);

    // Measure the end time
    clock_t end = clock();

    // Calculate the elapsed time in seconds
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Optionally, print the first few elements of the result vector Y
    printf("Resulting Y vector after SAXPY operation (first 10 elements):\n");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("Y[%d] = %f\n", i, Y[i]);
    }

    // Output the time taken for the SAXPY operation
    printf("\nTime taken for SAXPY operation: %f seconds\n", elapsed_time);

    // Clean up memory
    free(X);
    free(Y);

    return 0;
}

