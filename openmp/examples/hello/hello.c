#include <omp.h>
#include <stdio.h>
int main() {
  int id;
#pragma omp parallel private(id)
  {
    id = omp_get_thread_num();
    if (id%2==1)
      printf("Hello world from thread %d, I am odd\n", id);
    else
      printf("Hello world from thread %d, I am even\n", id);
  }
}
