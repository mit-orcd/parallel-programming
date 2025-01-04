#include <time.h>
#include <stdio.h>
#include <omp.h>

int main (void)
{

  int i;
  int n=1000000000;
  double sum,a[n];
  double tstart, tend;

  #pragma omp for private(i)
  for (i=0; i<n; i++) a[i]=(i+1)*0.1;

  sum = 0;
  tstart = omp_get_wtime();

     for (i=0; i<n; i++) 
     {
        sum = sum + a[i];
     }

  tend = omp_get_wtime();
  printf("Value of sum after parallel region: %f\n",sum);
  printf("Elapsed time: %f\n", tend-tstart);

}
