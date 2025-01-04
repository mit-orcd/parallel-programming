#include <omp.h>

int main() {

int i,n=16;
int a[n];

#pragma omp parallel shared(n,a) private(i)
{

   #pragma omp for
   for(i=0; i<n; i++) {
      a[i]=i+n;
      int id=omp_get_thread_num();
      printf("%d \t %d \t %d \n", id, i, a[i]);
   }

}

}

