#pragma once
#include <cstdio>
#include <future>

#ifdef DEBUG
  #define PRINTF printf
  #define COUT cout
#else
  #define PRINTF while(0) printf
  #define COUT   while(0) cout
#endif

float xor128();
template<class T> T min(T a, T b) { return (a<b)?a:b; }
//TODO: don't assume args are float (enable_if)
template<typename T, typename... Tail>
T min(T a, T b, Tail... tail){
  T tmp= min(a,b);
  return min(tmp, tail...);
}
//! Allows access to the underlying container in STL adapters like priority_queue
template <class Container>
class ContainerAccessor : public Container {
public:
    typedef typename Container::container_type container_type;
    container_type get_container() { return this->c; }
};

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

struct Parameters{
  Parameters(unsigned k, int overlapCubes, int procNum,
             float xCubeL, float yCubeL, float zCubeL, int xCubeArr,
             int yCubeArr, int zCubeArr, int xArrGl=1, int yArrGl=1, int zArrGl=1):
    k(k), xCubeL(xCubeL), yCubeL(yCubeL), zCubeL(zCubeL), xCubeArr(xCubeArr), yCubeArr(yCubeArr), 
    zCubeArr(zCubeArr), xArrGl(xArrGl),yArrGl(yArrGl),zArrGl(zArrGl), pageSize(yCubeArr*xCubeArr),
    xOverlap(overlapCubes*xCubeL/xCubeArr), yOverlap(overlapCubes*yCubeL/yCubeArr),
    zOverlap(overlapCubes*zCubeL/zCubeArr), ranks(new int[procNum]) {}

  Parameters(const Parameters&) =delete;
  const unsigned k;                           //!< Number of neighbors to return
  const float xCubeL, yCubeL, zCubeL;         //!< Cube length in each dimension
  const int xCubeArr, yCubeArr, zCubeArr;     //!< Number of Cubes in CubeArray in each dimension
  const int xArrGl,yArrGl,zArrGl;             //!< Number of CubeArrays in entire space
  const int pageSize;                         //!< Size of an "page" of all the columns and rows in CubeArray
  const float xOverlap, yOverlap, zOverlap;   //!< Percentage of overlap between CubeArrays
  
  const std::unique_ptr<int[]> ranks;
  int rank(Point3 pCd) const{
    return ranks[pCd.x+pCd.y*xArrGl+pCd.z*yArrGl*xArrGl];
  }
};

