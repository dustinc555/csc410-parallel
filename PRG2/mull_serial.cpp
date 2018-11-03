#include <bits/stdc++.h>

using namespace std;


void usage();

int main( int argc, char* argv[] )
{
  
  if ( argc != 3 )
    usage();

  const int RS = atoi(argv[1]);
  const int SIZE = RS * RS; // square matrix N by N
  bool random = argv[2][0] == 'r';
  vector<double> a(SIZE);
  vector<double> b(RS);
  vector<double> c(RS, 0);

  double lowerLimit = random ? 0 : 1;
  double upperLimit = random ? 100000 : 1;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);
  std::uniform_real_distribution<double> unif(lowerLimit,upperLimit);
  for (int i = 0; i < a.size(); i++)
    a[i] = unif(re);
  for (int i = 0; i < b.size(); i++)
    b[i] = unif(re);

  
  auto start = chrono::steady_clock::now();
  /* Perform multiply */
  for (int i = 0; i < SIZE; i++)
    a[i] *= b[i % RS];

  for (int i = 0; i < SIZE; i++)
    c[i / RS] += a[i];
 
  
  /*
  cout << "Result..."<< endl;
  for (int i = 0; i < RS; i++)
    cout << c[i] << " "; 
  */
  auto end = chrono::steady_clock::now();
  cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  return 0; 
}

void usage()
{
  printf("./main <N> <mode>\n");
  printf("mode: 1 to fill matrix and vector with all 1's.\n");
  printf("\tr for all random numbers.\n");
  exit(1);
}
