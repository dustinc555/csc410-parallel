#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>


__global__ matrixMul(double * arr, double * b, double * c, const int N, const int SIZE)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
}


const int INCORRECT_NUM_ARGS_ERROR = 1;

void usage();

int main( int argc, char* argv[] )
{

  if ( argc != 2 )
    usage();

  const int N = atoi(argv[1]);
  const int SIZE = N * N; // square matrix N by N  

  thrust::device_vector<double> a(SIZE, 1);
  thrust::device_vector<double> b(SIZE, 2);
  thrust::device_vector<double> c(SIZE, 0);

  double * d_a = thrust::raw_pointer_cast(&a[0]);
  double * d_b = thrust::raw_pointer_cast(&b[0]);
  double * d_c = thrust::raw_pointer_cast(&c[0]);

  int blocks = (SIZE/512+1); 
  dim3 threads(512); 
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N, SIZE);
  
  
  return 0;
} 


void usage()
{
  printf("./main <N>\n");
  exit(INCORRECT_NUM_ARGS_ERROR);
}

