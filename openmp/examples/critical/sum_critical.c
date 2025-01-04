#include <time.h>
#include <stdio.h>
#include <omp.h>

int main (void)
{

  int i,TID;
  int n=1000000000;
  double sum,sumLocal,a[n];
  double tstart, tend;

  #pragma omp parallel for private(i)
  for (i=0; i<n; i++) a[i]=(i+1)*0.1;

  sum = 0.;
  tstart = omp_get_wtime();

  #pragma omp parallel shared(n,a,sum) private(i,TID,sumLocal)
  {
     TID = omp_get_thread_num();
     sumLocal = 0.;

     #pragma omp for
     for (i=0; i<n; i++) sumLocal += a[i];

     #pragma omp critical (update_sum)
     {
        sum += sumLocal;
        //printf("TID=%d: sumLocal=%f sum = %f\n",TID,sumLocal,sum);
     }

  } /*-- End of parallel region --*/

  tend = omp_get_wtime();
  printf("Value of sum after parallel region: %f\n",sum);
  printf("Elapsed time: %f\n", tend-tstart);

}
