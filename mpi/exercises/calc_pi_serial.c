#include <math.h>

int main(argc,argv)
int argc;
char *argv[];
{
    //int n, myid, numprocs, i;
    int myid, numprocs, i;
    long long int n=20000000;
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x;

   // printf("Enter the number of intervals: (0 quits) ");
   // scanf("%d",&n);
  
	h   = 1.0 / (double) n;
	sum = 0.0;
	for (i = 1; i <= n; i++) {
	    x = h * ((double)i - 0.5);
	    sum += 4.0 / (1.0 + x*x);
	}
	pi = h * sum;
    
    
    printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));

}
