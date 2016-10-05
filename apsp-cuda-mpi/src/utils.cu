#include "utils.h"
#include "DPtr.cuh"
#include <cuda_runtime.h>

template<>
HPinPtr<int>::HPinPtr(const int N) {
  COUT<<"[HPinPtr]: Allocating\n";
  gpuErrchk(cudaHostAlloc(&data_, N*sizeof(int), cudaHostAllocDefault));
}
template<>
HPinPtr<int>::~HPinPtr() { cudaFreeHost(data_); }
