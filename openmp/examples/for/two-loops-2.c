#include <omp.h>

int main() {

int i,n=16;
int a[n], b[n];

#pragma omp parallel for shared(n,a) private(i)
  for (i=0; i<n; i++)  a[i] = i+1;
  //  there is an implied barrier

#pragma omp parallel for shared(n,a,b) private(i)
   for (i=0; i<n; i++)  b[i] = 2 * a[i];

    
for (i=0; i<n; i++)  printf("%d \t %d \t %d \n",i,a[i],b[i]);

}

