#pragma once
#include <cstdio>
#include <future>
#include <iostream>
#include <cmath>

#ifdef __DEBUG__
  #define PRINTF printf
  #define COUT std::cout
#else
  #define PRINTF while(0) printf
  #define COUT   while(0) std::cout
#endif

float xor128();
template<class T> T min(T a, T b) { return (a<b)?a:b; }
//TODO: don't assume args are float (enable_if)
template<typename T, typename... Tail>
T min(T a, T b, Tail... tail){
  T tmp= min(a,b);
  return min(tmp, tail...);
}
template<class T> struct lessPtr{
  constexpr bool operator()(const T a, const T b) const { return *a < *b; }
};

//! C compatible struct of 3 floats
struct Point3f{
  //Point3f(float x, float y, float z): x(x), y(y), z(z) {}
	//Point3f() =default;
  float x,y,z;
};
//! C compatible struct of 3 ints
struct Point3{
  //Point3(int x, int y, int z): x(x), y(y), z(z) {}
	//Point3() =default;
  int x,y,z;
};

//! Total number of Cubes per dim must be divisible by number of CubeArrays on same dim
struct Parameters{
  Parameters(unsigned k, int overlapCubes,
             int xCubeTot, int yCubeTot, int zCubeTot,
             int xArrGl=1, int yArrGl=1, int zArrGl=1):
    k(k),
    xCubeL(1.0/xCubeTot), yCubeL(1.0/yCubeTot), zCubeL(1.0/zCubeTot),
    xCubeArr(xCubeTot/xArrGl), yCubeArr(yCubeTot/yArrGl), zCubeArr(zCubeTot/zArrGl),
    xArrGl(xArrGl),yArrGl(yArrGl),zArrGl(zArrGl),
    pageSize(yCubeArr*xCubeArr),
    xOverlap((float)overlapCubes/xCubeArr), yOverlap((float)overlapCubes/yCubeArr),
    zOverlap((float)overlapCubes/zCubeArr), overlapCubes(overlapCubes) {}

  Parameters(const Parameters&) =delete;
  const unsigned k;                           //!< Number of neighbors to return
  const float xCubeL, yCubeL, zCubeL;         //!< Cube length in each dimension
  const int xCubeArr, yCubeArr, zCubeArr;     //!< Number of Cubes in CubeArray in each dimension
  const int xArrGl,yArrGl,zArrGl;             //!< Number of CubeArrays in entire space
  const int pageSize;                         //!< Size of an "page" of all the columns and rows in CubeArray
  const float xOverlap, yOverlap, zOverlap;   //!< Percentage of overlap between CubeArrays
  const int overlapCubes;
};

