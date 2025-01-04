#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16  // Size of the array to be sorted

// Helper function to print an array
void print_array(int *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

// Merge function used in merge sort
void merge(int *array, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = array[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = array[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            array[k++] = L[i++];
        } else {
            array[k++] = R[j++];
        }
    }

    while (i < n1) {
        array[k++] = L[i++];
    }

    while (j < n2) {
        array[k++] = R[j++];
    }
}

// Sequential merge sort function
void merge_sort(int *array, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        merge_sort(array, left, mid);
        merge_sort(array, mid + 1, right);
        merge(array, left, mid, right);
    }
}

// Parallel merge sort using MPI
void parallel_merge_sort(int *array, int n, int rank, int size) {
    int local_n = n / size;
    int *local_array = (int *)malloc(local_n * sizeof(int));

    // Scatter the array to all processes
    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort the local array
    merge_sort(local_array, 0, local_n - 1);

    // Gather the sorted subarrays at the root process
    MPI_Gather(local_array, local_n, MPI_INT, array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // The root process merges the sorted subarrays
        int *sorted = (int *)malloc(n * sizeof(int));
        int *indices = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            indices[i] = i * local_n;
        }

        for (int i = 0; i < n; i++) {
            int min_index = -1;
            int min_value = INT_MAX;
            for (int j = 0; j < size; j++) {
                if (indices[j] < (j + 1) * local_n && array[indices[j]] < min_value) {
                    min_value = array[indices[j]];
                    min_index = j;
                }
            }
            sorted[i] = min_value;
            indices[min_index]++;
        }

        // Copy the sorted array back to the original array
        for (int i = 0; i < n; i++) {
            array[i] = sorted[i];
        }

        free(sorted);
        free(indices);
    }

    free(local_array);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int array[N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Initialize the array with random values
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            array[i] = rand() % 100;
        }
        printf("Unsorted array:\n");
        print_array(array, N);
    }

    // Perform parallel merge sort
    parallel_merge_sort(array, N, rank, size);

    if (rank == 0) {
        printf("Sorted array:\n");
        print_array(array, N);
    }

    MPI_Finalize();
    return 0;
}

