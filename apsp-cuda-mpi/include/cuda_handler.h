#pragma once
#include <cuda_runtime_api.h>

class CUDAHandler{
public:
  CUDAHandler(const int prank){
    cudaGetDeviceCount(&devCount_);
    cudaSetDevice(prank%devCount_);
  }
private:
  int devCount_;
};
