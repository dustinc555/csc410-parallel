#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>

using namespace std;


void usage();

int main( int argc, char* argv[] )
{ 
  char mode = 'v';
  char values = '1';
  int RS = 0;
  switch ( argc )  
  {
    case 4:
      values = argv[3][0];
    case 3:
      mode = argv[2][0];
    case 2:
      RS = atoi(argv[1]);
      break;
    default:
      usage();
  }

  const int SIZE = RS * RS; // square matrix N by N 

  vector<double> a(SIZE, 0);
  vector<double> b(RS, 0);
  vector<double> c(RS, 0);

  /* if load mode, load vectors from file
  otherwise generate according to values rule. */
  if (values != 'l')
  {
    bool random = values == 'r';
    double lowerLimit = random ? 0 : 1;
    double upperLimit = random ? 10 : 1;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine re(seed);
    std::uniform_real_distribution<double> unif(lowerLimit,upperLimit);

    for (int i = 0; i < a.size(); i++)
      a[i] = unif(re);
    for (int i = 0; i < b.size(); i++)
      b[i] = unif(re);
  }
  else
  {
    // read in matrix and vector from file 
    ifstream myfile("input.txt");
    for (int i = 0; i < a.size(); i++)
       myfile >> a[i];
    for (int i = 0; i < b.size(); i++) 
       myfile >> b[i];
    myfile.close();
  }
    
  auto start = chrono::steady_clock::now();
  /* Perform multiply */
  for (int i = 0; i < SIZE; i++)
    a[i] *= b[i % RS];

  for (int i = 0; i < SIZE; i++)
    c[i / RS] += a[i];
  

  #ifdef DEBUG
  cout << endl;
  for (int i = 0; i < SIZE; i++)
    cout << a[i] << " ";

  cout << endl;
  for (int i = 0; i < RS; i++)
    cout << c[i] << " ";
  #endif


  if (mode != 't')
  {
      for (int i = 0; i < RS; i++)
      cout << fixed << setprecision(2) << c[i] << " ";
  }
  else // the mode is v
  {
    auto end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  }

  return 0; 
}

void usage()
{
  printf("./main <N> <mode> <values>\n");
  printf("<N> the row size\n");
  printf("<mode> v to print result, t to print calculation time.\n");
  printf("<values> 1 for all 1 values, r for all random, l to load from input.txt file.");
  exit(1);
}
