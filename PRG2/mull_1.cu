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
#include <fstream>
#include <string>
#include <iomanip>


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
  int a_index = blockIdx.x * blockDim.x + threadIdx.x;

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

using namespace std;
/**** MAIN ***********************/
/*********************************/
int main( int argc, char* argv[] )
{
 
  int N = 0;      // row size
  char mode = 'v'; // what to print
  int threads = 0;  // total amount of threads if 0 defaults to 512 per block
  char values = '1'; // what to fill vectors with
  switch ( argc )
  {
    case 5: 
      threads = atoi(argv[4]);
    case 4:
      values = argv[3][0];
    case 3:
      mode = argv[2][0];
    case 2:
      N = atoi(argv[1]); 
      break;
    default:
      usage();
  }
  
  
  
  const int SIZE = N * N; // square matrix N by N
  
  thrust::host_vector<double> h_a(SIZE);
  thrust::host_vector<double> h_b(N);
  thrust::device_vector<double> d_a(SIZE, 1);
  thrust::device_vector<double> d_b(N, 1);
  thrust::device_vector<double> c(N);

  // if mode is load, load vectors from file, otherwise generate them ourselves
  if (values != 'l')
  {
    bool random = values == 'r';
    double lowerlimit = random ? 0 : 1;
    double upperlimit = random ? 3 : 1;
    #ifdef DEBUG
    printf("upperLimit: %f  lowerLimit: %f\n", upperlimit, lowerlimit);
    #endif
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine re(seed);
    std::uniform_real_distribution<double> unif(lowerlimit,upperlimit);
    for (int i = 0; i < SIZE; i++)
      h_a[i] = floor(unif(re));
    for (int i = 0; i < N; i++)
      h_b[i] = floor(unif(re));
  }
  else // load vectors from file
  { 
    ifstream myfile("input.txt");
    for (int i = 0; i < SIZE; i++)
       myfile >> h_a[i];
    for (int i = 0; i < N; i++)
       myfile >> h_b[i];
    myfile.close();
  }
  
  /* thrust handles the copying of memory from host vectors to
     device vectors with a simple assignment. */  
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
  // if we were given a set amount of threads
  // set to it
  if ( threads )
  {
     #ifdef DEBUG
     if (N > threads)
       cout << "Warning! incorrect number of threads will not perform correctly." << endl;
     #endif
 
     blocks = threads; // ensures that there are exactly as many given threads on the problem
     threads = 1;
  }
  else
  {
    threads = THREADS;
    blocks = (SIZE / THREADS) + 1;
  }

  #ifdef DEBUG
    cout << "blocks: " << blocks << " threads: " << threads << endl;
  #endif

  // record action time 
  auto start = chrono::steady_clock::now();
 

  performMults<<<blocks, threads>>>(p_a, p_b, N, SIZE);
  cudaDeviceSynchronize(); 
  sumRows<<<blocks, threads>>>(p_a, p_c, N, SIZE);
  cudaDeviceSynchronize();

  
  auto end = chrono::steady_clock::now();
  
  // print out time took if requested
  if (mode == 't')
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  
  thrust::host_vector<double> result = c;

  #ifdef DEBUG
  printf("\n\nresult:\n");
  #endif
  
  if (mode == 'v')
    for (int i = 0; i < N; i++)
      cout << fixed << setprecision(2) << result[i] << " ";

  #ifdef DEBUG 
  cout << endl;
  #endif
   
  return 0;
} 


void usage()
{
  printf("./main <row size> <mode> <values> <threads>\n");
  printf("<row size> : required\n<mode> : v to print result, t to print time nanoseconds\n<values> : 1 all 1 values, r all random, l load from file.\n");
  exit(INCORRECT_NUM_ARGS_ERROR);
}




