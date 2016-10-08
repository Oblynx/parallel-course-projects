#include <cstdio>
#include <cstring>
#include <memory>
#include <chrono>
#include "utils.h"
#include "mpi_handler.h"
#include "tasks.h"

using namespace std;


// MPI node main functions
int master(MPIhandler& mpi, int argc, char** argv);
extern int slave(MPIhandler& mpi, int argc, char** argv);

int main(int argc, char** argv){
  MPIhandler mpi(true, &argc, &argv);
  auto status= (mpi.rank())?
                  slave(mpi, argc,argv): 
                  master(mpi, argc,argv);
  return status;
}

int master(MPIhandler& mpi, int argc, char** argv){
  FILE* fin= stdin;
  int inSpecified= 0;
  for(int i=1; i<argc; i++) if(!strcmp(argv[i],"-i")) inSpecified= i;
  if(inSpecified) fin= fopen(argv[inSpecified+1], "r");
  else printf("Reading from stdin\n");
  if (fin==NULL){
    printf("Wrong input file\n");
    exit(3);
  }
  FILE* logfile= stdout;
#ifdef LOG
  int logSpecified= 0;
  for(int i=1; i<argc; i++) if(!strcmp(argv[i],"-l")) logSpecified= i;
  if(!logSpecified){
    printf("Logging mode enabled. To run, specify logfile path as command line argument:\n\
            Use: %s [-i <inputfile>] -l <logfile>\n", argv[0]);
    exit(2);
  }
  logfile= fopen(argv[logSpecified+1], "a");
  if(logfile==NULL){
    printf("Wrong log file\n");
    exit(4);
  }
#endif
#ifdef NO_TEST
  printf("WARNING! No_TEST has been defined\n");
#endif
  
  // Input data
  int N;
  while(!fscanf(fin, "%d\n", &N));
  N= 1<<N;
  unique_ptr<int[]> g(new int[N*N]);
  unique_ptr<int[]> groundTruth(new int[N*N]), groundTruthGPU(new int[N*N]);
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      while(!fscanf(fin, "%d", &g[i*N+j]));
  printf("N=%d\n", N);
#ifdef LOG
  fprintf(logfile, "%d;", N);
#endif
  
  // Run algorithms
#ifndef NO_TEST
  //run_cpu_test(g.get(),N, groundTruth.get(), logfile);
  run_gpu_test(g.get(),N, groundTruthGPU.get(), logfile);
#endif
  run_gpu_mpi_master(mpi, g.get(),N, groundTruth.get(), logfile);

#ifdef LOG
  fprintf(logfile, "\n");
  fclose(logfile);
#endif
  auto check= test(g.get(), groundTruth.get(), N, "mpi");
  if(check){
    printf("\t***Test SUCCESSFUL!***\n");
  }else{
    printf("\t***Test FAILED!***\n");
    exit(1);
  }
  return 0;
}

