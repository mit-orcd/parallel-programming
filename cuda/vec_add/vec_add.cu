#include <stdio.h>

__global__ void add(int *a, int *b, int *c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

#define N (4096*4096)
#define THREADS_PER_BLOCK 512

int main()
{
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = N * sizeof( int );

/* get GPU device number and name */

  int dev;
  cudaDeviceProp deviceProp;
  cudaGetDevice( &dev );
  cudaGetDeviceProperties( &deviceProp, dev );
  printf("Using GPU %d: %s\n", dev, deviceProp.name );

/* allocate space for device copies of a, b, c */

  cudaMalloc( (void **) &d_a, size );
  cudaMalloc( (void **) &d_b, size );
  cudaMalloc( (void **) &d_c, size );

/* allocate space for host copies of a, b, c and setup input values */

  a = (int *)malloc( size );
  b = (int *)malloc( size );
  c = (int *)malloc( size );

  for( int i = 0; i < N; i++ )
  {
    a[i] = b[i] = i;
    c[i] = 0;
  }

/* copy inputs to device */

  cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

/* launch the kernel on the GPU */

  add<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );

/* copy result back to host */

  cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );

  int success = 1;

  for( int i = 0; i < N; i++ )
  {
    if( c[i] != a[i] + b[i] )
    {
      printf("c[%d] = %d\n",i,c[i] );
      success = 0;
      break;
    } /* end if */
  }

  if( success == 1 ) printf("PASS\n");
  else               printf("FAIL\n");

/* clean up */

  free(a);
  free(b);
  free(c);
  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );

  cudaDeviceReset();
	
  return 0;
} /* end main */
