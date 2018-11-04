#include <bits/stdc++.h>

using namespace std;


void usage();

int main( int argc, char* argv[] )
{
  
  if ( argc != 3 )
    usage();

  const int RS = atoi(argv[1]);
  const int SIZE = RS * RS; // square matrix N by N 

  #ifndef TEST
  bool random = argv[2][0] == 'r';
  #endif

  vector<double> a(SIZE, 0);
  vector<double> b(RS, 0);
  vector<double> c(RS, 0);


  // if not test build, intialize vectors else read them from file.
  #ifndef TEST
  double lowerLimit = random ? 0 : 1;
  double upperLimit = random ? 100000 : 1;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);
  std::uniform_real_distribution<double> unif(lowerLimit,upperLimit);

  for (int i = 0; i < a.size(); i++)
    a[i] = unif(re);
  for (int i = 0; i < b.size(); i++)
    b[i] = unif(re);
  #else
  // read in matrix and vector from file 
  ifstream myfile("input.txt");
  for (int i = 0; i < a.size(); i++)
     myfile >> a[i];
  for (int i = 0; i < b.size(); i++) 
     myfile >> b[i];
  myfile.close();
  #endif
    
  auto start = chrono::steady_clock::now();
  /* Perform multiply */
  for (int i = 0; i < SIZE; i++)
    a[i] *= b[i % RS];

  for (int i = 0; i < SIZE; i++)
    c[i / RS] += a[i];
  /*
  cout << endl;
  for (int i = 0; i < SIZE; i++)
    cout << a[i] << " ";

  cout << endl;
  for (int i = 0; i < RS; i++)
    cout << c[i] << " ";*/


  #ifndef TIMED
    for (int i = 0; i < RS; i++)
    cout << fixed << setprecision(2) << c[i] << " ";
  #else
  auto end = chrono::steady_clock::now();
  cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  #endif
  return 0; 
}

void usage()
{
  printf("./main <N> <mode>\n");
  printf("mode: 1 to fill matrix and vector with all 1's.\n");
  printf("\tr for all random numbers.\n");
  exit(1);
}
