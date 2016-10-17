#pragma once
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include "utils.h"

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
  //! 1D allocation
  DPtr(const int N): Nx_(N), Ny_(1), pitch_(N) {
    gpuErrchk(cudaMalloc( (void**)&data_, N*sizeof(T) ));
  }
  //! 2D allocation
  DPtr(const int Nx, const int Ny): Nx_(Nx), Ny_(Ny) {
    gpuErrchk(cudaMallocPitch( (void**)&data_, &pitch_, Nx*sizeof(T), Ny ));
  }
  ~DPtr() { cudaFree(data_); }

  //! 1D copies
  void copyH2D (T* a, const int N, const int offset= 0){
    gpuErrchk( cudaMemcpy(data_+offset, a, N*sizeof(T), cudaMemcpyHostToDevice) );
  }
  void copyD2H (T* a, const int N, const int offset= 0){
    gpuErrchk( cudaMemcpy(a, data_+offset, N*sizeof(T), cudaMemcpyDeviceToHost) );
  }

  //! Multi 1D copies
  // DANGER: Won't work if Nx > Nx_ - offset (won't wrap around to the next row because of the pitch)
  void copyH2D_multi (T* a, const int Nx, const int Ny, const xy offset) {
    if(Nx_ - offset.x > Nx_) throw new std::invalid_argument("Offset.x must be less for the requested row elements\n:");
    if(Ny_ - offset.y > Ny_) throw new std::invalid_argument("Offset.y must be less for the requested rows\n");
    for(int row=0; row<Ny; row++)
      gpuErrchk( cudaMemcpy((T*)((char*)data_+ pitch_*(row+offset.y))+offset.x, a+Nx*row,
            Nx*sizeof(T), cudaMemcpyHostToDevice) );
  }
  void copyD2H_multi (T* a, const int Nx, const int Ny, const xy offset) {
    if(Nx_ - offset.x > Nx_) throw new std::invalid_argument("Offset.x must be less for the requested row elements\n:");
    if(Ny_ - offset.y > Ny_) throw new std::invalid_argument("Offset.y must be less for the requested rows\n");
    for(int row=0; row<Ny; row++){
      printf("[D2H]: N=(%d,%d)\toff=(%d,%d)\n",Nx,Ny, offset.x,offset.y);
      gpuErrchk( cudaMemcpy(a+Nx*row, (T*)((char*)data_+ pitch_*(row+offset.y))+offset.x,
            Nx*sizeof(T), cudaMemcpyDeviceToHost) );
    }
  }

  //! 2D copies
  void copyH2D (T* a, const int pitch_elt, const int Nx, const int Ny, const xy offset= xy(0,0)) {
    gpuErrchk(cudaMemcpy2D( (T*)((char*)data_+ pitch_*offset.y)+ offset.x, pitch_,
          a, pitch_elt*sizeof(T), Nx*sizeof(T), Ny, cudaMemcpyHostToDevice ));
  }
  void copyD2H (T* a, const int pitch_elt, const int Nx, const int Ny, const xy offset=xy(0,0)) {
    gpuErrchk(cudaMemcpy2D(a, pitch_elt*sizeof(T), (T*)((char*)data_+pitch_*offset.y)+offset.x,
          pitch_, Nx*sizeof(T), Ny, cudaMemcpyDeviceToHost));
  }


  int pitch_elt() const { return pitch_/sizeof(T); }
  T* get() const { return data_; }
  operator T*() const { return data_; }
 private:
  const int Nx_, Ny_;
  size_t pitch_;
  T* data_;
};


/*template<class T>
struct DPtr{
  DPtr(const int N): N_(N) {
    gpuErrchk(cudaMalloc( (void**)&data_, N*sizeof(T) ));
  }
  void copyH2D (T* a, const int N, const int offset= 0){
    gpuErrchk( cudaMemcpy(data_+offset, a, N*sizeof(T), cudaMemcpyHostToDevice) );
  }
  void copyD2H (T* a, const int N, const int offset= 0){
    gpuErrchk( cudaMemcpy(a, data_+offset, N*sizeof(T), cudaMemcpyDeviceToHost) );
  }
  //! Multi 1D copies
  void copyH2D_multi (T* a, const int Nx, const int Ny, const int offset, const int stride= -1) {
    if(stride<0) stride= N_;
    for(int i=0; i<Ny; i++)
      gpuErrchk( cudaMemcpy(data_+offset+ stride*i, a+Nx*i, Nx*sizeof(T), cudaMemcpyHostToDevice) );
  }
  void copyD2H_multi (T* a, const int Nx, const int Ny, const int offset, const int stride= -1) {
    if(stride<0) stride= N_;
    for(int i=0; i<Ny; i++)
      gpuErrchk( cudaMemcpy(a+Nx*i, data_+offset+ stride*i, Nx*sizeof(T), cudaMemcpyDeviceToHost) );
  }

  int pitch_elt() const { return N_; }
 private:
  const int N_;
  T* data_;
};*/

//! Pinned host memory pointer for quick PCIe transfers
template<class T>
struct HPinPtr{
    HPinPtr(): owns(false) {}
    HPinPtr(const int N): owns(true) { alloc(N); }
    ~HPinPtr() { dealloc(); }
    T& operator[](size_t i) const { return data_[i]; }
    operator T*() const { return data_; }
    T* get() const { return data_; } 
    void reset(int N) {
      if(owns) dealloc();
      alloc(N);
      owns= true;
    }
    T* operator+(int n) const { return data_+n; }
  private:
    void alloc(const int N) {
      gpuErrchk(cudaHostAlloc( (void**)&data_, N*sizeof(int), cudaHostAllocDefault ));
    }
    void dealloc() { cudaFreeHost(data_); }

    bool owns;
    T* data_;
};

class CUDAHandler{
public:
  CUDAHandler(const int prank){
    cudaGetDeviceCount(&devCount_);
    cudaSetDevice(prank%devCount_);
  }
  void synchronize() { gpuErrchk(cudaStreamSynchronize(cudaStreamPerThread)); }
private:
  int devCount_;
};
