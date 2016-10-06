#pragma once
#include <cstdio>
#include <memory>
#include "utils.h"
#include "mpi_handler.h"

// CPU Floyd-Warshall
Duration_fsec run_cpu(const int* g, const int N, std::unique_ptr<int[]>& result_cpu, FILE* logfile);

// GPU block algo
Duration_fsec run_GPUblock(MPIhandler mpi, const int* g, const int N, const std::unique_ptr<int[]>& groundTruth,
    FILE* logfile );

// GPU block algo -- multiple vertices per thread (y only)
Duration_fsec run_GPUblock_multiy(MPIhandler mpi, const HPinPtr<int>& g, const int N, const std::unique_ptr<int[]>& groundTruth,
    FILE* logfile );

