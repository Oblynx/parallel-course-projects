#pragma once
#include <cstdio>
#include <memory>
#include "utils.h"
#include "mpi_handler.h"

// Tests
double run_cpu_test(const int* g, const int N, int* result_cpu, FILE* logfile);
double run_gpu_test(const int* g, const int N, int* result_gpu, FILE* logfile);

// MPI algo
double run_gpu_mpi_master(MPIhandler& mpi, int* g, const int N, const int* groundTruth, FILE* logfile);
