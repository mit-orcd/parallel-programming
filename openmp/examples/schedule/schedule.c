#include <stdio.h>
#include <omp.h>

int main (void)
{

  int i,j,n=16;

  #pragma omp parallel for default(none) schedule(runtime) private(i,j) shared(n)
//  #pragma omp parallel for default(none) schedule(static,2) private(i,j) shared(n)
  for (i=0; i<n; i++){
      printf("Iteration %d executed by thread %d\n", i, omp_get_thread_num());
      for (j=0; j<i; j++)   system("sleep 0.5");
  } 

}
