#include <stdio.h>
#include <omp.h>

int main (void)
{
//    int i = 10;
    int i, vtest=10, n=4;

//    #pragma omp parallel for private(i,vtest) shared(n)
//    #pragma omp parallel for private(i) firstprivate(vtest) shared(n)
//    #pragma omp parallel for private(i) lastprivate(vtest) shared(n)
    #pragma omp parallel for private(i) firstprivate(vtest) lastprivate(vtest) shared(n)
    for(i=0; i<n; i++)
    {
        printf("thread %d: initial value = %d\n", omp_get_thread_num(), vtest);
        vtest=i;
        //vtest = omp_get_thread_num();
        //printf("thread %d: final value = %d\n", omp_get_thread_num(), vtest);
    }

    printf("value after loop = %d\n", vtest);

    return 0;
}
