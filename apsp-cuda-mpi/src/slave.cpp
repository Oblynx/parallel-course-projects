#include <chrono>
#include <iostream>
#include <cuda_runtime_api.h>
#include "mpi_handler.h"
#include "utils.h"
#include "DPtr.h"
#include "kernel_wrp.h"

using namespace std;

void run_gpu_mpi_slave(MPIhandler& mpi, int N);

int slave(MPIhandler& mpi, int argc, char** argv){
  int N;
  mpi.bcast(&N,1);
  run_gpu_mpi_slave(mpi, N);
  return 0;
}

void run_gpu_mpi_slave(MPIhandler& mpi, int N){
  // Constants
  constexpr const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(n,n);
  if(N<MAX_THRperBLK2D) bs= dim3(N,N);
  
  // MPI transfer setup
  mpi.makeTypes(n,N);
  mpi.splitMat(N);
  const int s_x= mpi.submatRowL(), s_y= mpi.submatRowN();

  // Allocate GPU memory
  DPtr<int> d_g2(2*n*N);
  DPtr<int> d_g3(s_x*s_y);
  unique_ptr<int[]> msgRowcol(new int[2*n*N]);
  unique_ptr<int[]> msgSubmat(new int[s_x*s_y]);
  for(int b=0; b<B; b++){
    mpi.scatterMat(nullptr, msgSubmat.get());
    mpi.bcast(msgRowcol.get(), 2*n*N); 

    d_g2.copy(msgRowcol.get(),2*n*N, Dir::H2D);
    d_g3.copy(msgSubmat.get(),s_x*s_y, Dir::H2D);
    const int yStart= (mpi.submatStart()/n)/B, xStart= (mpi.submatStart()/n)%B;
    phase3(dim3(s_x/n-1, s_y/n-1),bs, d_g3, d_g2, b,N, xStart,yStart, s_x);
    d_g3.copy(msgSubmat.get(),s_x*s_y, Dir::D2H);
    cudaStreamSynchronize(cudaStreamPerThread);
    PRINTF("[run_gpu]: b=%d phase3 complete\n",b);

    PRINTF("[run_gpu]: MPI msg:\n");
    for(int i=0; i<s_y; i++){
      for(int j=0; j<s_x; j++){
        printf("%3d ", msgSubmat[i*s_x+j]);
      }
      printf("\n");
    }
    printf("\n");

    mpi.gatherMat(msgSubmat.get(), nullptr);
    PRINTF("[run_gpu]: b=%d matrix gathered\n",b);
    mpi.barrier();
  }
}
