#pragma once
#include "mpi_handler.h"
#include "cuda_handler.h"
#include "DPtr.h"

//! Compute APSP in g; for rank!=0, g is NULL
double floydWarshall_gpu_mpi(int *g, int N, MPIHandler& mpi, CUDAHandler& cuda);

//! Main algorithm loop.
// dsg: device submat of g belonging to this process
void loopTiles(int* dsg, const int B, const int N, MPIHandler& mpi, CUDAHandler& cuda);

//! Decide if this proc takes part in phase
int phase1FindExecutor(const int b, MPIHandler& mpi);
int phase2RowFindExecutor(const int b, const int col, MPIHandler& mpi);
int phase2ColFindExecutor(const int b, const int row, MPIHandler& mpi);

//! Execute phase
void execPhase1 (DPtr<int>& dsg, const int b, const int N, int* tilebuf, DPtr<int>& d_tile, MPIHandler& mpi);
void execPhase2Row(DPtr<int>& dsg, const int b, const int N, DPtr<int>& d_tile, int* rowbuf,
    DPtr<int>& d_row, MPIHandler& mpi);
void execPhase3 (DPtr<int>& dsg, const int b, const int N, MPIHandler& mpi);
