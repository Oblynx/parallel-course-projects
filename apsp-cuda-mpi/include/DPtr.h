#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

//! Manages pitched (2D) data on the GPU
template<class T>
struct DPtr{
  DPtr(const int Nx, const int Ny): Nx(Nx), Ny(Ny) {
    gpuErrchk(cudaMallocPitch(&data_, &pitch_, Nx*sizeof(T), Ny));
  }
  ~DPtr() { cudaFree(data_); }
  void copyH2D (T* a, const int pitch_elt, const int Nx, const int Ny, const int offset= 0) {
    gpuErrchk(cudaMemcpy2D(data_, pitch_, a+offset, pitch_elt*sizeof(T), Nx*sizeof(T), Ny, cudaMemcpyHostToDevice));
  }
  void copyD2H (T* a, const int pitch_elt, const int Nx, const int Ny, const int offset= 0) {
    gpuErrchk(cudaMemcpy2D(a+offset, pitch_elt*sizeof(T), data_, pitch_, Nx*sizeof(T), Ny, cudaMemcpyDeviceToHost));
  }
  int pitch_elt() const { return pitch_/sizeof(T); }
  T* get() const { return data_; }
  operator T*() const { return data_; }
 private:
  const int Nx, Ny;
  size_t pitch_;
  T* data_;
};

