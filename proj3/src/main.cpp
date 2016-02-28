#include <cstdio>
#include <cstring>
#include <memory>
#include <chrono>
#include "utils.h"
using namespace std;

extern Duration_fsec run_cpu(const HPinPtr<int>& g, const int N, unique_ptr<int[]>& result_cpu, FILE* logfile);
extern Duration_fsec run_GPUsimple(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile);
extern Duration_fsec run_GPUblock(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile);
extern Duration_fsec run_GPUblock_multixy(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile);
extern Duration_fsec run_GPUblock_multiy(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile);

int main(int argc, char** argv){
  FILE* fin= stdin;
  if(argc>2 && !strcmp(argv[1],"-i")) fin= fopen(argv[2], "r");
  else if(argc>4 && strcmp(argv[3],"-i")) fin= fopen(argv[4], "r");
  if (fin==NULL){
    printf("Wrong input file\n");
    exit(3);
  }
  FILE* logfile= stdout;
#ifdef LOG
  if(argc<2 || (!strcmp(argv[1],"-l") && (argc<4 || !strcmp(argv[3], "-l")))){
    printf("Logging mode enabled. To run, specify logfile path as command line argument:\nUse: %s -l <logfile>\n", argv[0]);
    exit(2);
  }
  int l_idx= (argc==3)? 2: 4;
  logfile= fopen(argv[l_idx], "a");
  if(logfile==NULL){
    printf("Wrong log file\n");
    exit(4);
  }
#endif
#ifdef NO_TEST
  printf("WARNING! No_TEST has been defined\n");
#endif

  int N;
  while(!fscanf(fin, "%d\n", &N));
  HPinPtr<int> g(N*N);
  unique_ptr<int[]> groundTruth(new int[N*N]);
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      while(!fscanf(fin, "%d", &g[i*N+j]));
  printf("\nN=%d\n", N);
#ifdef LOG
  fprintf(logfile, "%d;", N);
#endif
  
#ifndef NO_TEST
  run_cpu(g,N, groundTruth, logfile);
#endif
  run_GPUsimple(g,N, groundTruth, logfile);
  run_GPUblock(g,N, groundTruth, logfile);
  run_GPUblock_multixy(g,N, groundTruth, logfile);
  run_GPUblock_multiy(g,N, groundTruth, logfile);

#ifdef LOG
  fprintf(logfile, "\n");
  fclose(logfile);
#endif
  return 0;
}

