#include "utils.h"
#include "DPtr.h"
#include <cuda_runtime.h>

template<>
HPinPtr<int>::HPinPtr(const int N) {
  gpuErrchk(cudaHostAlloc(&data_, N*sizeof(int), cudaHostAllocDefault));
}
template<>
HPinPtr<int>::~HPinPtr() { cudaFreeHost(data_); }
