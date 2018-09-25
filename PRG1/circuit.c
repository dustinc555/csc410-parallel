/* File:    circuit.c
 * Author:  Dustin Cook
 * Purpose: Try's every combination of inputs for the circuit.
 *
 * Input:   thread_count
 * Output:  Successful input combinations, the thread that made the successful combination,
 *          and the timeing.
 *
 * Compile: gcc -g -Wall -fopenmp -o circuit circuit.c
 * Usage:   ./main <number of threads> <>
 *
 * Notes: There does appear to be a small performance boost when using dynamic instead of static of about 4.7%.
 *
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


int check_circuit (int id, int z);
void Usage(char* prog_name);

int main(int argc, char *argv[])
{
  double averageTime = 0.0;
  const unsigned int MAX_VALUE = 65535;
  unsigned int i;
  int thread_count;
  int count = 0;
  double start = omp_get_wtime();
  if (argc != 2) Usage(argv[0]);
  thread_count = strtol(argv[1], NULL, 10);

#   pragma omp parallel for num_threads(thread_count) schedule( static, 1 ) reduction(+:count, averageTime)
    for (i = 0; i <= MAX_VALUE; i++)
    {
      double wtime = omp_get_wtime();
       if (check_circuit(omp_get_thread_num(), i))
         count += 1;
       wtime = omp_get_wtime() - wtime;
       averageTime += wtime;
    }
    // barrier already exists
  # pragma omp single
  {
    averageTime /= MAX_VALUE;
    printf("count: %d\naverage time per thread: %f\n", count, averageTime);
    printf("program time: %f\n", omp_get_wtime() - start);
  }

  return 0;
}


/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
 #define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)
 /* Check if a given input produces TRUE (a one) */
 int check_circuit (int id, int z)
 {
  int v[16]; /* Each element is a bit of z */
  int i;
   for (i = 0; i < 16; i++) v[i] = EXTRACT_BIT(z,i);
   if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3])
     && (!v[3] || !v[4]) && (v[4] || !v[5])
     && (v[5] || !v[6]) && (v[5] || v[6])
     && (v[6] || !v[15]) && (v[7] || !v[8])
     && (!v[7] || !v[13]) && (v[8] || v[9])
     && (v[8] || !v[9]) && (!v[9] || !v[10])
     && (v[9] || v[11]) && (v[10] || v[11])
     && (v[12] || v[13]) && (v[13] || !v[14])
     && (v[14] || v[15])) {
     printf ("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,
     v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],
     v[10],v[11],v[12],v[13],v[14],v[15]);
     fflush (stdout);
     return 1;
   } else return 0;
 }

 /*--------------------------------------------------------------------
 * Function:    Usage
 * Purpose:     Print command line for function and terminate
 * In arg:      prog_name
 */
void Usage(char* prog_name)
{
   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   exit(0);
}  /* Usage */
