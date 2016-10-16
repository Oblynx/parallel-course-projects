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

#define MAX_THRperBLK2D 4
#define MAX_THRperBLK2D_MULTI 32

struct xy{
  xy(const int x=0, const int y=0): x(x),y(y) {}
  xy operator+(const xy& other) const { return xy(x+other.x, y+other.y); }
  xy operator-(const xy& other) const { return xy(x-other.x, y-other.y); }
  xy operator*(const int n) const { return xy(x*n, y*n); }
  xy operator/(const int n) const { return xy(x/n, y/n); }
  xy operator*(const xy& oth) const { return xy(x*oth.x, y*oth.y); }
  xy operator/(const xy& oth) const { return xy(x/oth.x, y/oth.y); }
  bool operator==(const xy& other) const { return (x==other.x)&&(y==other.y); }
  int x, y;
};

//! Smart pointer singleton
template<class T>
class smart_ptr{
  public:
    smart_ptr(bool make= true): owns(false), data_(NULL) { 
      if(make) owns= true, data_= new T;
    }
    smart_ptr(const smart_ptr&);              // delete copy
    smart_ptr& operator=(const smart_ptr&);   // delete copy
    ~smart_ptr() {
      if(owns) delete(data_);
    }

    T& operator*() const { return *data_; }
    operator T*() const { return data_; }
    T* get() const { return data_; }
    void reset(int N) {
      if(owns) delete(data_);
      data_= new T[N];
      owns= true;
    }
    T* operator+(int n) const { return data_+n; }
  private:
    bool owns;
    T* data_;
};

//! Smart pointer for handling dynamic arrays without c++11
template<class T>
class smart_arr{
  public:
    smart_arr(): owns(false), data_(NULL) {}
    smart_arr(int N): owns(true) { data_= new T[N]; }
    smart_arr(const smart_arr&);              // delete copy
    smart_arr& operator=(const smart_arr&);   // delete copy
    ~smart_arr() {
      if(owns) delete[](data_);
    }
    T& operator[](int i) const { return data_[i]; }
    T* get() const { return data_; }
    T* operator+(int n) const { return data_+n; }
    void reset(int N) {
      if(owns) delete[](data_);
      data_= new T[N];
      owns= true;
    }
  private:
    bool owns;
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

