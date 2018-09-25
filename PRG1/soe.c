/* File:    soe.c
 * Author:  Dustin Cook
 * Purpose: Finds all primes up to given n.
 *
 * Input:   n
 * Output:  All primes leading up to n.
 *
 * Compile: gcc -g -Wall -fopenmp -o soe soe.c -lm
 * Usage:   ./soe <n> <Print Table: 1 for yes, 0 for no>
 *
 * Notes: There does appear to be a significant performance boost when using
 *        dynamic vs static up to 46%. For the average thread time, i only take account for threads that do work eg: they have found a prime
          and they have to find the multiples because otherwise it would throw off the average time.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define false 0
#define true 1


void Usage(char* prog_name);
void printDivider();
void printPrimeTable(char * marks, int size, int n);


int main(int argc, char *argv[])
{
  int n;
  int p;
  int longRuns = 0;
  int printTable = 0;
  double start = omp_get_wtime();
  double averageTime = 0.0;
  int thread_count = 8;
  n = strtol(argv[1], NULL, 10);
  printTable = strtol(argv[2], NULL, 10);
  int size = n + 1;
  char marks[size];

  marks[0] = marks[1] = false;
  for (int i = 2; i < size; i++) {
    marks[i] = true;
  }

  printf("n: %d, array size: %d, thread_count: %d\n", n, size, thread_count);

  # pragma omp parallel for num_threads(thread_count) schedule( static, 1 ) reduction(+:averageTime)
  for (p = 2; p < size; p++) {
    int local_p = false;
    # pragma omp critical (validateP)
    {
      if (marks[p] == true)
        local_p = p;
    }
    if (local_p != false) {
      for (int i = local_p * 2; i <= size; i += local_p) {
           marks[i] = false;
      }
    }

    if (local_p != false) {
      averageTime += omp_get_wtime() - start;
      longRuns++;
    }
  }

  # pragma omp single
  {
    if (printTable == true)
      printPrimeTable(marks, size, n);
    else {
      averageTime /= longRuns;
      printf("average thread time: %f\ntotal time: %f\n", averageTime, omp_get_wtime() - start);
    }
  }


  return 0;
}


 /*--------------------------------------------------------------------
 * Function:    Usage
 * Purpose:     Print command line for function and terminate
 * In arg:      prog_name
 */
void Usage(char* prog_name)
{
   fprintf(stderr, "usage: %s <n> <Print Input: 1 for yes, 0 for no>\n", prog_name);
   exit(0);
}  /* Usage */

void printPrimeTable(char * marks, int size, int n) {
  printf("Prime List | %d", n);
  printDivider();
  printf("%d: ", 0);
  for (int i = 0; i < size; i++) {
    if ((i + 1) % 50 == 0) {
      printDivider();
      printf("%d's Range: ", i+1);
    }
    if (marks[i] == true)
      printf("%d | ", i);
  }
  printDivider();
}

void printDivider() {
  printf("\n--------------------------------------------------------------------------------\n");
}
