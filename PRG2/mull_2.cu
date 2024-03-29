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

__global__ void performMults(double * a, double * b, int ROW_SIZE, int SIZE)
{
  int a_index = blockIdx.x * blockDim.x + threadIdx.x;
  int b_index = a_index % ROW_SIZE;

  if (a_index >= SIZE) return;
  // The multiplication stage must be done before the mapping and reduction stage
  // all of these tasks can be done in parallel
  a[a_index] *= b[b_index];

}

/** matrixMul(double * arr, double * b, double * c, const int N, const int SIZE)
*   Expects arr to be a matrix, b a vector, and c a result vector
*   c[i] = sum(a[i,j] * b[i])
*  
*/
__global__ void matrixMul(double * a, double * b, double * c, int ROW_SIZE, int SIZE)
{


  int a_index = blockIdx.x * blockDim.x + threadIdx.x;
  int b_index = a_index % ROW_SIZE;
  int c_index = a_index / ROW_SIZE;

  int offset = c_index * ROW_SIZE; // the row we are working with
  //a[a_index] = a_index;
  
  // Reduction stage
  // sum up the local array and place it into its according c_index
  for (int s = 1; s < SIZE; s *= 2) 
  {
    int index = 2 * s * b_index;
    if (index + s < ROW_SIZE) 
      a[index + offset] += a[index + offset + s];
    __syncthreads();
  }
  
  if (b_index == 0)
    c[c_index] = a[offset];
 
}

const int INCORRECT_NUM_ARGS_ERROR = 1;

void printVector(thrust::device_vector<double> a);
void usage();
void fillVector(thrust::host_vector<double> & vec, bool allOnes);


/**** MAIN ***********************/
/*********************************/
int main( int argc, char* argv[] )
{
  #ifdef DEBUG
  auto start = chrono::steady_clock::now();
  #endif

  if ( argc != 3 )
    usage();

  const int N = atoi(argv[1]);
  const int SIZE = N * N; // square matrix N by N

  thrust::host_vector<double> h_a(SIZE);
  thrust::host_vector<double> h_b(N);
  thrust::device_vector<double> d_a(SIZE);
  thrust::device_vector<double> d_b(N);
  thrust::device_vector<double> c(N);
   
  bool random = argv[2][0] == 'r';

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
  
  // keep threads below 1024 but ensure no partial rows... hmm
  
  int blocks = N;
  int threads = N;
  //cout << "blocks: " << blocks << " threads: " << THREADS << endl;
  performMults<<<blocks, threads>>>(p_a, p_b, N, SIZE);
  cudaDeviceSynchronize();
  matrixMul<<<blocks, threads>>>(p_a, p_b, p_c, N, SIZE);
  cudaDeviceSynchronize();

  thrust::host_vector<double> result = c;
  h_a = d_a;
  
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
  
  #ifdef DEBUG
  cout << "Reduction result on matrix:" << endl;
  for (int i = 0; i < SIZE; i++)
  {
    cout << h_a[i] << " ";
    if ((i + 1) % N == 0) cout << endl;
  }
  #endif
  
  #ifdef DEBUG
  auto end = chrono::steady_clock::now();
  cout << "Elapsed time in nanoseconds: "
        << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
        << " ns" << endl;
  #endif
  
  return 0;
} 


void usage()
{
  printf("./main <N> <mode>\n");
  printf("mode: 1 to fill matrix and vector with all 1's.\n");
  printf("\tr for all random numbers.\n");
  exit(INCORRECT_NUM_ARGS_ERROR);
}




