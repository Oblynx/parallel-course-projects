#pragma once
#include <memory>
#include <cstdio>
#include <iostream>
#include <cassert>

#ifdef __DEBUG__
  #define PRINTF printf
  #define COUT std::cout
#else
  #define PRINTF while(0) printf
  #define COUT   while(0) std::cout
#endif

#define MAX_THRperBLK2D 4
#define MAX_THRperBLK2D_MULTI 32

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

struct HPinPtr{
  HPinPtr(const int N);
  ~HPinPtr();
  int& operator[](size_t i) const { return data_[i]; }
  operator int*() const { return data_; }
  int* get() const { return data_; } 
  private:
  int* data_;
};

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

//Debug
/*
inline void printG(const int* g, const int n, const int Nx, int Ny= -1){
  assert(n<=Nx);
  if(Ny==-1) Ny= Nx;
  int idx= 0;
  for(int i=0; i<Ny; i++){
    for(int j=0; j<Nx; j++){
      if((j+1)%n) idx+= printf("%3d ", g[i*Nx+j]);
      else    idx+= printf("%3d|", g[i*Nx+j]);
    }
    if((i+1)%n) idx+= printf("\n");
    else{
      idx+= printf("\n");
      for(int j=0; j<Nx; j++)
        idx+= printf("----");
      idx+= printf("\n");
    }
  }
  idx+= printf("_____________________________________\n");
}
*/
inline void printG(const int* g, const int n, const int Nx, int Ny= -1){
  assert(n<=Nx);
  if(Ny==-1) Ny= Nx;
  smart_arr<char> buf(7*Nx*Ny);
  int idx= 0;
  for(int i=0; i<Ny; i++){
    for(int j=0; j<Nx; j++){
      if((j+1)%n) idx+= sprintf(buf+idx,"%3d ", g[i*Nx+j]);
      else    idx+= sprintf(buf+idx,"%3d|", g[i*Nx+j]);
    }
    if((i+1)%n) idx+= sprintf(buf+idx,"\n");
    else{
      idx+= sprintf(buf+idx,"\n");
      for(int j=0; j<Nx; j++)
        idx+= sprintf(buf+idx,"----");
      idx+= sprintf(buf+idx,"\n");
    }
  }
  idx+= sprintf(buf+idx,"_____________________________________\n");
  printf("%s\n",buf.get());
}
