#include <cuda_runtime_api.h>

class CUDAHandler{
  CUDAHandler(const int prank){
    cudaGetDeviceCount(&devCount_);
    cudaSetDevice(prank%devCount_);
  }
private:
  int devCount_;
};
