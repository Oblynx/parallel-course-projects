#pragma once
#include <memory>
#include <chrono>
#include <cstdio>
#include <iostream>

#ifdef __DEBUG__
  #define PRINTF printf
  #define COUT std::cout
#else
  #define PRINTF while(0) printf
  #define COUT   while(0) std::cout
#endif

template<class T>
struct HPinPtr{
  HPinPtr(const int N);
  ~HPinPtr();
  T& operator[](size_t i) const { return data_[i]; }
  operator T*() const { return data_; }
  T* get() const { return data_; } 
  private:
  T* data_;
};

typedef std::chrono::duration<float, std::ratio<1>> Duration_fsec;

inline bool test(const HPinPtr<int>& toCheck, const std::unique_ptr<int[]>& truth, const int N, std::string name){
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
inline void printG(const std::unique_ptr<int[]>& g, const int N){
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++)
      printf("%3d\t", g[i*N+j]);
    printf("\n");
  }
  printf("_____________________________________\n");
}


