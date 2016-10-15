#include <cstdio>
#include "utils.h"
#include "algo.h"

int main(int argc, char** argv){
  MPIHandler mpi(&argc, &argv);
  CUDAHandler cuda(mpi.rank());
  smart_ptr<smart_arr<int>> g(false), truth(false);
  int N;
  if(!mpi.rank()){
    N= input((*g).get());
    double testTime= (N<512)?
      run_cpu_test((*g).get(), N, (*truth).get()):
      run_gpu_test((*g).get(), N, (*truth).get());
  }
  mpi.bcast(&N,1);
  
  if(!mpi.rank()) printf("Starting GPU MPI fw\n");
  double time= floydWarshall_gpu_mpi((*g).get(), N, mpi, cuda);
  if(!mpi.rank()) printf("fw time: %.3f\n", time);

  if(!mpi.rank()) test(g, truth, "gpu_mpi");
  return 0;
}
