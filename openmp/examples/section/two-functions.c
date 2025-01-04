#include<omp.h>

void funcA(const int n, double a[]){
 
  int i; 
  for (i=1; i<n-1; i++)  a[i] = ( a[i-1] + a[i+1] ) / 2.;

}

void funcB(const int n, double b[]){
 
  int i; 
  for (i=1; i<n; i++)  b[i] = (b[i-1] + 1.)/2.;

}


int main() {

  int i,n=600000000;
  double a[n], b[n];

// initialize
  for (i=0; i<n; i++){
    if(i==0){
       a[i]=1.;
       b[i]=1.;
    } else if(i==n-1){
       a[i]=n*3.;
    } else{
       a[i]=0.;
    }
  }

#pragma omp parallel sections
{
  #pragma omp section
  funcA(n, a);

  #pragma omp section
  funcB(n, b);
}   
 
  //for (i=0; i<n; i++)  printf("%d \t %f \t %f \n", i, a[i], b[i]);

}

