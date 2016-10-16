#include <cstdio>
#include <cstring>
#include "utils.h"
#include "algo.h"
#include "test.h"

int input(smart_arr<int>& g, int argc, char** argv);

int main(int argc, char** argv){
  MPIHandler mpi(&argc, &argv);
  CUDAHandler cuda(mpi.rank());
  smart_arr<int> g, truth;
  int N;
  if(!mpi.rank()){
    N= input(g, argc, argv);
    if (N<512) run_cpu_test(g.get(), N, truth.get());
    else       run_gpu_test(g.get(), N, truth.get());
  }
  mpi.bcast(&N,1,0);
  
  if(!mpi.rank()) printf("Starting GPU MPI fw\n");
  double time= floydWarshall_gpu_mpi(g.get(), N, mpi, cuda);
  if(!mpi.rank()) printf("fw time: %.3f\n", time);

  if(!mpi.rank()) test(g.get(), truth.get(), N, "gpu_mpi");
  return 0;
}

int input(smart_arr<int>& g, int argc, char** argv){
  FILE* fin= NULL;
  int inSpecified= 0;
  for(int i=1; i<argc; i++) if(!strcmp(argv[i],"-i")) inSpecified= i;
  if(inSpecified) fin= fopen(argv[inSpecified+1], "r");
  if (fin==NULL){
    printf("No input file detected\nSyntax: ./apsp -i <in_file>\n");
    return(3);
  }
  #ifdef NO_TEST
    printf("WARNING! No_TEST has been defined\n");
  #endif
  
  // Input data
  int N;
  while(!fscanf(fin, "%d\n", &N));
  N= 1<<N;
  g.reset(N*N);
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      while(!fscanf(fin, "%d", &g[i*N+j]));
  printf("N=%d\n", N);

  return N;
}
