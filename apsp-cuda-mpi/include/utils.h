#pragma once
#include <memory>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <cuda_runtime_api.h>

#ifdef __DEBUG__
  #define PRINTF printf
  #define COUT std::cout
#else
  #define PRINTF while(0) printf
  #define COUT   while(0) std::cout
#endif

#define MAX_THRperBLK2D 16
#define MAX_THRperBLK2D_MULTI 32

struct xy{
  xy(const int x, const int y): x(x),y(y) {}
  const int x, y;
};

//! Pinned host memory pointer for quick PCIe transfers
template<class T>
struct HPinPtr{
    HPinPtr(const int N) { gpuErrchk(cudaHostAlloc(&data_, N*sizeof(int), cudaHostAllocDefault)); }
    ~HPinPtr() { cudaFreeHost(data_); }
    T& operator[](size_t i) const { return data_[i]; }
    operator T*() const { return data_; }
    T* get() const { return data_; } 
  private:
    T* data_;
};

//! Smart pointer for handling dynamic arrays without c++11
template<class T>
class smart_arr{
  public:
    smart_arr(int N) { data_= new T[N]; }
    smart_arr(const smart_arr&);
    smart_arr& operator=(const smart_arr&);
    ~smart_arr() { delete[](data_); }
    T& operator[](int i) { return data_[i]; }
    T* get() { return data_; }
    T* operator+(int n) { return data_+n; }
  private:
    T* data_;
};


//! Test if arrays are equal
inline bool test(const int* toCheck, const int* truth, const int N, std::string name){
  #ifndef NO_TEST
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      if(toCheck[i*N+j] != truth[i*N+j]){
        printf("[test/%s]: Error at (%d,%d)! toCheck/truth =\n\t%d\n\t%d\n", name.c_str(), i,j, toCheck[i*N+j],
            truth[i*N+j]);
        return false;
      }
  #endif
  return true;
}

//! Print array g for debugging purposes 
inline void printG(const int* g, const int n, const int Nx, const int Ny_= -1){
  assert(n<=Nx);
  const int Ny= (Ny_<0)? Nx: Ny_;
  smart_arr<char> buf(7*Nx*Ny);

  int idx= 0;
  for(int i=0; i<Ny; i++){
    for(int j=0; j<Nx; j++){
      if((j+1)%n) idx+= sprintf(buf+idx,"%3d ", g[i*Nx+j]);
      else        idx+= sprintf(buf+idx,"%3d|", g[i*Nx+j]);
    }
    idx+= sprintf(buf+idx,"\n");
    if(!(i+1)%n){
      for(int j=0; j<Nx; j++)
        idx+= sprintf(buf+idx,"----");
      idx+= sprintf(buf+idx,"\n");
    }
  }
  idx+= sprintf(buf+idx,"_____________________________________\n");
  printf("%s\n",buf.get());
}


















