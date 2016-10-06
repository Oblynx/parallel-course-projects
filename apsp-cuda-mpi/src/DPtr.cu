#include "DPtr.h"

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

DPtr::DPtr(int N) { gpuErrchk(cudaMalloc(&data_, N*sizeof(int))); }
DPtr::~DPtr() { cudaFree(data_); }
void DPtr::copy(int* a, const int N, const Dir dir, const devOffset=0) {
  if(dir == Dir::H2D) gpuErrchk(cudaMemcpy(data_+devOffset, a, sizeof(int)*N, cudaMemcpyHostToDevice));
  else gpuErrchk(cudaMemcpy(a, data_, sizeof(int)*N, cudaMemcpyDeviceToHost));
}

int* DPtr::get() const { return data_; }
DPtr::operator int*() const { return data_; }

