#pragma once
#include <memory>
#include <cstdio>
#include <iostream>

#ifdef __DEBUG__
  #define PRINTF printf
  #define COUT std::cout
#else
  #define PRINTF while(0) printf
  #define COUT   while(0) std::cout
#endif

#define MAX_THRperBLK2D 32
#define MAX_THRperBLK2D_MULTI 32

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
inline void printG(const int* g, const int N, const int n){
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      if((j+1)%n) printf("%3d ", g[i*N+j]);
      else    printf("%3d|", g[i*N+j]);
    }
    if((i+1)%n) printf("\n");
    else{
      printf("\n");
      for(int j=0; j<N; j++)
        printf("----");
      printf("\n");
    }
  }
  printf("_____________________________________\n");
}

