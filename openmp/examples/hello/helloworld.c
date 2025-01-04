#include <omp.h>

int main() {

#pragma omp parallel
{
   printf("Hello world!\n");
}

}

