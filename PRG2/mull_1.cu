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


__global__ void performMults(double * a, double * b, int ROW_SIZE, int SIZE)
{
  int a_index = blockIdx.x * blockDim.x + threadIdx.x;

  int b_index = a_index % ROW_SIZE;

  if (a_index >= SIZE) return;
  // The multiplication stage must be done before the mapping and reduction stage
  // all of these tasks can be done in parallel
  a[a_index] *= b[b_index];

}


using namespace std;

/** sumRows(double * arr, double * b, double * c, const int N, const int SIZE)
*   Expects arr to be a matrix, and c a result vector
*   c[i] = sum(a[i,j] * b[i])
*  
*/
__global__ void sumRows(double * a, double * c, const int ROW_SIZE, const int SIZE)
{
  #ifndef TIMED
  int a_index = blockIdx.x * blockDim.x + threadIdx.x;
  #else
  int a_index = blockIdx.x;
  #endif

  int b_index = a_index % ROW_SIZE; // you can consider b_index the row id (0 start, ROW_SIZE-1 end)
  
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
const unsigned THREADS = 512;

void usage();


/**** MAIN ***********************/
/*********************************/
int main( int argc, char* argv[] )
{
  if ( argc < 3 )
    usage();
 
  unsigned threads = THREADS;
  const int N = atoi(argv[1]);
  const int SIZE = N * N; // square matrix N by N

  thrust::host_vector<double> h_a(SIZE);
  thrust::host_vector<double> h_b(N);
  thrust::device_vector<double> d_a(SIZE, 1);
  thrust::device_vector<double> d_b(N, 1);
  thrust::device_vector<double> c(N);
  
  #ifndef TIMED 
  bool random = argv[2][0] == 'r';
  #else
  bool random = argv[3][0] == 'r';
  threads = atoi(argv[2]);
  #endif

  double lowerLimit = random ? 0 : 1;
  double upperLimit = random ? 3 : 1;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

  #ifdef DEBUG
  printf("upperLimit: %f  lowerLimit: %f\n", upperLimit, lowerLimit);
  #endif

  std::default_random_engine re(seed);
  std::uniform_real_distribution<double> unif(lowerLimit,upperLimit);
  for (int i = 0; i < h_a.size(); i++)
    h_a[i] = floor(unif(re));
  for (int i = 0; i < h_b.size(); i++)
    h_b[i] = floor(unif(re));
  

  d_a = h_a;
  d_b = h_b;
  
  #ifdef DEBUG
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
  #endif

  // vectors are unfortunatly not available on cuda device
  // but you can get the memory address, pass it to the device,
  // and treat it as a normal array.
  double * p_a = thrust::raw_pointer_cast(&d_a[0]);
  double * p_b = thrust::raw_pointer_cast(&d_b[0]);
  double * p_c = thrust::raw_pointer_cast(&c[0]);

  unsigned blocks;
  // one thread per block, if doing the Karp-Flatt Metric
  #ifdef TIMED
  blocks = threads;
  threads = 1;
  #else
  // just make sure that there are enough threads
  blocks = (SIZE / threads) + 1;
  #endif


  // record action time 
  #ifdef TIMED
  auto start = chrono::steady_clock::now();
  #endif

  performMults<<<blocks, threads>>>(p_a, p_b, N, SIZE);
  cudaDeviceSynchronize(); 
  sumRows<<<blocks, threads>>>(p_a, p_c, N, SIZE);
  cudaDeviceSynchronize();

  #ifdef TIMED
  auto end = chrono::steady_clock::now();
  cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  #endif


  thrust::host_vector<double> result = c;

  #ifdef DEBUG
  printf("\n\nresult:\n");
  #endif

  #ifndef TIMED
  for (int i = 0; i < result.size(); i++)
    cout << result[i] << " ";
  #endif

  #ifdef DEBUG 
  cout << endl;
  #endif
   
  return 0;
} 


void usage()
{
  printf("./main <N> <mode>\n");
  printf("mode: 1 to fill matrix and vector with all 1's.\n");
  printf("\tr for all random numbers.\n");
  printf("if make Timed: ./main <N> <threads> <mode>\n");
  exit(INCORRECT_NUM_ARGS_ERROR);
}




