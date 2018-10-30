/** Thrust Library **/
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/** Std library **/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <time.h>
#include <chrono>


using namespace std;

/** matrixMul(double * arr, double * b, double * c, const int N, const int SIZE)
*   Expects arr to be a matrix, b a vector, and c a result vector
*   c[i] = sum(a[i,j] * b[i])
*  
*/
__global__ void matrixMul(double * a, double * b, double * c, const int ROW_SIZE, const int SIZE)
{
  int a_index = blockIdx.x * blockDim.x + threadIdx.x;
  int b_index = a_index % ROW_SIZE;

  if (a_index >= SIZE) return;

  // create the multiplies, we will sum them later
  a[a_index] *= b[b_index];
  __syncthreads();
  
  if (b_index == 0) // if we are a zero index, sum up the row up to but not including the next 0 row.
  {
    int local_c_sum = 0;
    for (int i = 0; i < ROW_SIZE; i++)
      local_c_sum += a[a_index + i];

    int c_index = a_index / ROW_SIZE;
    c[c_index] = local_c_sum; 
  }
  // this method is bad because its tasks size grow with the problem instead of the number of tasks. 
}

const int INCORRECT_NUM_ARGS_ERROR = 1;
const int THREADS = 512;

void printVector(thrust::device_vector<double> a);
void usage();
void fillVector(thrust::host_vector<double> & vec, bool allOnes);


/**** MAIN ***********************/
/*********************************/
int main( int argc, char* argv[] )
{
  if ( argc != 3 )
    usage();

  const int N = atoi(argv[1]);
  const int SIZE = N * N; // square matrix N by N

  thrust::host_vector<double> h_a(SIZE);
  thrust::host_vector<double> h_b(N);
  thrust::device_vector<double> d_a(SIZE, 1);
  thrust::device_vector<double> d_b(N, 1);
  thrust::device_vector<double> c(N);
   
  bool random = argv[2][0] == 'r';
  printf("random: %d\n", random);

  double lowerLimit = random ? 0 : 1;
  double upperLimit = random ? 5 : 1;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

  printf("upperLimit: %f  lowerLimit: %f\n", upperLimit, lowerLimit);
  std::default_random_engine re(seed);
  std::uniform_real_distribution<double> unif(lowerLimit,upperLimit);
  for (int i = 0; i < h_a.size(); i++)
    h_a[i] = unif(re);
  for (int i = 0; i < h_b.size(); i++)
    h_b[i] = unif(re);
  

  d_a = h_a;
  d_b = h_b;

  cout << "Matrix values:" << endl;
  for (int i = 0; i < SIZE; i++) 
  {
    cout << h_a[i] << " ";
    if ((i + 1) % N == 0) cout << endl;
  }
  cout << "\n\n";
  cout << "Vector values:" << endl;
  for (int i = 0; i < N; i++)
    cout << h_b[i] << " ";
  cout << endl;


  // vectors are unfortunatly not available on cuda device
  // but you can get the memory address, pass it to the device,
  // and treat it as a normal array.
  double * p_a = thrust::raw_pointer_cast(&d_a[0]);
  double * p_b = thrust::raw_pointer_cast(&d_b[0]);
  double * p_c = thrust::raw_pointer_cast(&c[0]);

  int blocks = (SIZE/THREADS+1); 
  
  matrixMul<<<blocks, THREADS>>>(p_a, p_b, p_c, N, SIZE);

  cudaDeviceSynchronize();

  thrust::host_vector<double> result = c;

  printf("\n\nresult:\n");
  for (int i = 0; i < result.size(); i++)
    cout << "result[" << i << "] = " << result[i] << endl; 
  
  return 0;
} 


void usage()
{
  printf("./main <N> <mode>\n");
  printf("mode: 1 to fill matrix and vector with all 1's.\n");
  printf("\tr for all random numbers.\n");
  exit(INCORRECT_NUM_ARGS_ERROR);
}




