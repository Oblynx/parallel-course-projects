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
  DPtr<int> d_g2(2*n*N);
  DPtr<int> d_g3(s_x*s_y);
  int* msgRowcol= new int[2*n*N];
  int* msgSubmat= new int[s_x*s_y];
  for(int b=0; b<B; b++){
    mpi.scatterMat(NULL, msgSubmat);
    mpi.bcast(msgRowcol, 2*n*N); 

    d_g2.copy(msgRowcol,2*n*N, 0);
    d_g3.copy(msgSubmat,s_x*s_y, 0);
    const int yStart= (mpi.submatStart()/n)/B, xStart= (mpi.submatStart()/n)%B;
    phase3(dim3(s_x/n-1, s_y/n-1),bs, d_g3, d_g2, b,N, xStart,yStart, s_x);
    d_g3.copy(msgSubmat,s_x*s_y, 1);
    cudaDeviceSynchronize();
    PRINTF("[run_gpu]: b=%d phase3 complete\n",b);

    PRINTF("[run_gpu]: MPI msg:\n");
    for(int i=0; i<s_y; i++){
      for(int j=0; j<s_x; j++){
        printf("%3d ", msgSubmat[i*s_x+j]);
      }
      printf("\n");
    }
    printf("\n");

    mpi.gatherMat(msgSubmat, NULL);
    PRINTF("[run_gpu]: b=%d matrix gathered\n",b);
    mpi.barrier();
  }
  delete[](msgRowcol); delete[](msgSubmat);
}
