#include "utils.h"
#include "DPtr.h"
#include <cuda_runtime.h>

HPinPtr::HPinPtr(const int N) {
  gpuErrchk(cudaHostAlloc(&data_, N*sizeof(int), cudaHostAllocDefault));
}
HPinPtr::~HPinPtr() { cudaFreeHost(data_); }
