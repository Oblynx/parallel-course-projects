#include "mpi_handler.h"
#include "cuda_handler.h"
#include "DPtr.h"

//! Compute APSP in g; for rank!=0, g is NULL
double floydWarshall_gpu_mpi(int *g, int N, MPIHandler& mpi, CUDAHandler& cuda);

//! Main algorithm loop.
// dsg: device submat of g belonging to this process
void loopTiles(DPtr<int>& dsg, const int B, MPIHandler& mpi, CUDAHandler& cuda);

bool phase1ExecCheck(const int b, const xy gridCd);
bool phase2ExecCheck(const int b, const xy gridCd);

void execPhase1(DPtr<int>& dsg, MPIHandler& mpi);
void execPhase2(DPtr<int>& dsg, MPIHandler& mpi);
void execPhase3(DPtr<int>& dsg, MPIHandler& mpi);
