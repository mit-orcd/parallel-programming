#include <time.h>
#include <stdio.h>
#include <omp.h>

int main (void)
{

  int TID;
  time_t t1,t2;

  #pragma omp parallel private(TID,t1)
  {
    TID = omp_get_thread_num();
    if (TID < omp_get_num_threads()/2 ) system("sleep 10");
    t1 =  time(NULL);
    printf("Thread %d before barrier at %s \n", omp_get_thread_num(), ctime(&t1) );

    #pragma omp barrier

    t2 =  time(NULL);
    printf("Thread %d after barrier at %s \n", omp_get_thread_num(), ctime(&t2) );
  } /*-- End of parallel region --*/

}
