#include <stdio.h>

void add_vectors(const int a[], const int b[], int c[], int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int size;

    // Prompt user for the size of the vectors
    printf("Enter the size of the vectors: ");
    scanf("%d", &size);

    int a[size], b[size], c[size];

    // Initialize vector a
    printf("Enter elements of vector a:\n");
    for (int i = 0; i < size; i++) {
        scanf("%d", &a[i]);
    }

    // Initialize vector b
    printf("Enter elements of vector b:\n");
    for (int i = 0; i < size; i++) {
        scanf("%d", &b[i]);
    }

    // Perform vector addition
    add_vectors(a, b, c, size);

    // Display the result
    printf("Result of vector addition (c = a + b):\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    return 0;
}

