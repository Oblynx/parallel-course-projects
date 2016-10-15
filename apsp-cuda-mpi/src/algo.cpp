#include <ctime>
#include "utils.h"
#include "DPtr.h"
#include "mpi_handler.h"
#include "cuda_handler.h"
#include "algo.h"

using namespace std;

#define n MAX_THRperBLK2D

//! Compute APSP in g; for rank!=0, g is NULL
double floydWarshall_gpu_mpi(int *g, int N, MPIHandler& mpi, CUDAHandler& cuda){
  const int B= N/n;
  mpi.makeGrid(N);
  DPtr<int> dsubmat(mpi.s_x()*mpi.s_y());

  double begin= clock();
  mpi.scatterMat(g, dsubmat);
  loopTiles(dsubmat, B, N, mpi,cuda);
  mpi.gatherMat(dsubmat, g);
  double end= clock() - begin;

  return end;
}

void loopTiles(DPtr<int> dsg, const int B, const int N, MPIHandler& mpi, CUDAHandler& cuda){
  for(int b=0; b<B; b++){
    if(phase1ExecCheck(b, mpi.gridCoord()))
      execPhase1(dsg, b,N, mpi);
    if(phase2ExecCheck(b, mpi.gridCoord()))
      execPhase2(dsg, b,N, mpi);
    execPhase3(dsg, b,N, mpi);
  }
}


void execPhase1(DPtr<int> dsg, const int b, const int N, MPIHandler& mpi){
  const dim3 bs(n,n);
  phase1(1,bs, dsg, b,N);
  mpi.bcast;
}
