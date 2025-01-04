#include <omp.h>

int main() {

int i,n=16;
int a[n], b[n];

#pragma omp parallel shared(n,a,b) private(i)
{
    #pragma omp for
    for (i=0; i<n; i++)  a[i] = i+1;
    //  there is an implied barrier

    #pragma omp for
    for (i=0; i<n; i++)  b[i] = 2 * a[i];

}      /*-- End of parallel region --*/
    
for (i=0; i<n; i++)  printf("%d \t %d \t %d \n",i,a[i],b[i]);

}

