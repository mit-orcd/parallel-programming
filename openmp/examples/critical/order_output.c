#include <time.h>
#include <stdio.h>
#include <omp.h>

int main (void)
{

  int TID;

  #pragma omp parallel private(TID) 
  {

     TID = omp_get_thread_num();
     #pragma omp critical (print_tid)  
     {
        printf("Thread %d : Hello, ",TID);
        printf("world!\n");
     }
  }  /*-- End of parallel region --*/

}
