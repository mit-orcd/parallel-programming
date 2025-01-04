#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  int i;
  long long int n=100000000;
  clock_t start_time, end_time;
  double x, pi;
  double sum = 0.0;
  double step = 1.0/(double) n; 

  start_time = clock();
  /* Parallelize the following block of code */
  for (i = 0; i < n; i++) {
    x = (i+0.5)*step;
    sum = sum + 4.0/(1.0+x*x);
  }
  pi = step * sum;
  end_time = clock();

  printf("pi = %17.15f\n",pi);
  printf("time to compute = %g seconds\n", (double)(end_time - start_time)/CLOCKS_PER_SEC);
  return 0;
}
