#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiply_matrices(float** a, float** b, float** c, int size) {
    // Initialize result matrix to zero
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[i][j] = 0.0f;
        }
    }

    // Perform matrix multiplication
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    int size = 20480;  // Size of the square matrices

    // Allocate memory for matrices using pointers
    float** a = (float**)malloc(size * sizeof(float*));
    float** b = (float**)malloc(size * sizeof(float*));
    float** c = (float**)malloc(size * sizeof(float*));

    for (int i = 0; i < size; i++) {
        a[i] = (float*)malloc(size * sizeof(float));
        b[i] = (float*)malloc(size * sizeof(float));
        c[i] = (float*)malloc(size * sizeof(float));
    }

    // Seed the random number generator
    srand(time(NULL));

    // Generate random elements for matrix A
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i][j] = (float)(rand() % 100) / 10.0f;  // Random float numbers between 0.0 and 9.9
        }
    }

    // Generate random elements for matrix B
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            b[i][j] = (float)(rand() % 100) / 10.0f;  // Random float numbers between 0.0 and 9.9
        }
    }

    // Measure the start time
    clock_t start_time = clock();

    // Perform matrix multiplication
    multiply_matrices(a, b, c, size);

    // Measure the end time
    clock_t end_time = clock();

    // Calculate the time taken
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Display the time taken
    printf("Time taken for multiplication: %f seconds\n", time_taken);

    // Free allocated memory
    for (int i = 0; i < size; i++) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    return 0;
}

