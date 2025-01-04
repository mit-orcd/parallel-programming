#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to swap two elements
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Partition function used in quicksort
int partition(int *array, int low, int high) {
    int pivot = array[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (array[j] <= pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }
    swap(&array[i + 1], &array[high]);
    return i + 1;
}

// Sequential quicksort function
void quicksort(int *array, int low, int high) {
    if (low < high) {
        int pi = partition(array, low, high);
        quicksort(array, low, pi - 1);
        quicksort(array, pi + 1, high);
    }
}

// Helper function to merge two sorted arrays
void merge(int *result, int *left, int left_size, int *right, int right_size) {
    int i = 0, j = 0, k = 0;
    while (i < left_size && j < right_size) {
        if (left[i] <= right[j]) {
            result[k++] = left[i++];
        } else {
            result[k++] = right[j++];
        }
    }
    while (i < left_size) {
        result[k++] = left[i++];
    }
    while (j < right_size) {
        result[k++] = right[j++];
    }
}

// Parallel quicksort function using MPI
void parallel_quicksort(int *array, int n, int rank, int size) {
    int local_n = n / size;
    int *local_array = (int *)malloc(local_n * sizeof(int));

    // Scatter the array to all processes
    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort the local array
    quicksort(local_array, 0, local_n - 1);

    // Gather the sorted subarrays at the root process
    int *sorted = NULL;
    if (rank == 0) {
        sorted = (int *)malloc(n * sizeof(int));
    }
    MPI_Gather(local_array, local_n, MPI_INT, sorted, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Merge the sorted subarrays at the root process
        int *temp = (int *)malloc(n * sizeof(int));
        merge(temp, sorted, local_n, sorted + local_n, local_n);
        for (int i = 2 * local_n; i < n; i += local_n) {
            merge(sorted, temp, i, sorted + i, local_n);
            int *swap_temp = temp;
            temp = sorted;
            sorted = swap_temp;
        }
        if (temp != sorted) {
            for (int i = 0; i < n; i++) {
                sorted[i] = temp[i];
            }
        }
        free(temp);
    }

    // Copy the sorted array back to the original array
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            array[i] = sorted[i];
        }
        free(sorted);
    }

    free(local_array);
}

// Helper function to print an array
void print_array(int *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    int array[16];
    int n = 16;  // Size of the array to be sorted

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Initialize the array with random values
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            array[i] = rand() % 100;
        }
        printf("Unsorted array:\n");
        print_array(array, n);
    }

    // Perform parallel quicksort
    parallel_quicksort(array, n, rank, size);

    if (rank == 0) {
        printf("Sorted array:\n");
        print_array(array, n);
    }

    MPI_Finalize();
    return 0;
}

