#include <iostream>
#include <cuda_runtime_api.h>
#include "mpi_handler.h"
#include "utils.h"
#include "DPtr.h"
#include "kernel_wrp.h"

using namespace std;

void run_gpu_mpi_slave(MPIhandler& mpi, int N);

int slave(MPIhandler& mpi, int argc, char** argv){
  cudaSetDevice(mpi.rank());
  int N;
  mpi.bcast(&N,1);
  run_gpu_mpi_slave(mpi, N);
  return 0;
}

void run_gpu_mpi_slave(MPIhandler& mpi, int N){
  // Constants
  const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(n,n);
  if(N<MAX_THRperBLK2D) bs= dim3(N,N);
  
  // MPI transfer setup
  mpi.splitMat(N);
  mpi.makeTypes(n,N);
  const int s_x= mpi.submatRowL(), s_y= mpi.submatRowN();

  // Allocate GPU memory
  DPtr<int> d_rowcol(2*n*N);
  DPtr<int> d_submat(s_x*s_y);
  int* msgRowcol= new int[2*n*N];
  int* msgSubmat= new int[s_x*s_y];
  mpi.scatterMat(NULL, msgSubmat);
  d_submat.copy(msgSubmat,s_x*s_y, 0);
  for(int b=0; b<B; b++){
    mpi.bcast(msgRowcol, 2*n*N); 
    mpi.barrier();
    d_rowcol.copy(msgRowcol,2*n*N, 0);

    const int yStart= (mpi.submatStart()/n)/B, xStart= (mpi.submatStart()/n)%B, xEnd= B-xStart;
    updateSubmat(2*B-yStart-xStart,bs, d_submat, d_rowcol, b,N, xEnd,xStart,yStart,
                 mpi.submatRowL()/n,mpi.submatRowN()/n);
    phase3(dim3(s_x/n-1, s_y/n-1),bs, d_submat, d_rowcol, b,N, xStart,yStart, s_x);
    PRINTF("[run_gpu]: b=%d phase3 complete\n",b);
  }
  d_submat.copy(msgSubmat,s_x*s_y, 1);
  cudaDeviceSynchronize();
  mpi.gatherMat(msgSubmat, NULL);
  mpi.barrier();
  PRINTF("[run_gpu]: matrix gathered\n");
  delete[](msgRowcol); delete[](msgSubmat);
}
