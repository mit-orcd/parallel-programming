#include <stdio.h>
#include <time.h>

int main() {
  int i;
  long long int n=100000000;
  float a=2.0;
  float x[n];
  float y[n];
  clock_t start_time, end_time;

  /* Parallelize this block of code (optional) */
  for (i = 0; i < n; i++){
    x[i] = 1.0;
    y[i] = 2.0;
  }

  start_time = clock();
  /* Parallelize this block of code */
  for (i = 0; i < n; i++){
    y[i] = a*x[i] + y[i];
  }
  end_time = clock();
  printf ("SAXPY Time: %f\n",(double)(end_time - start_time)/CLOCKS_PER_SEC);
  
}
