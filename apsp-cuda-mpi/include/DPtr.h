#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

template<class T>
struct DPtr{
  DPtr(int N) { gpuErrchk(cudaMalloc(&data_, N*sizeof(T))); }
  ~DPtr() { cudaFree(data_); }
  void copy (T* a, const int N, const int dir, const int offset=0) {
    if(dir == 0) gpuErrchk(cudaMemcpy(data_+offset, a, sizeof(T)*N, cudaMemcpyHostToDevice));
    else gpuErrchk(cudaMemcpy(a, data_+offset, sizeof(T)*N, cudaMemcpyDeviceToHost));
  }
  T* get() const { return data_; }
  operator T*() const { return data_; }
  private:
  T* data_;
};

