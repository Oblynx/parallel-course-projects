#pragma once
#include <cstdio>
//#include <mpi.h>

#ifdef DEBUG
  #define PRINTF printf
  #define COUT cout
#else
  #define PRINTF while(0) printf
  #define COUT   while(0) cout
#endif

double xor128();
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

struct Point3d{
  Point3d(double x, double y, double z): x(x), y(y), z(z) {}
	Point3d() =default;
  double x,y,z;
};
struct Point3{
  Point3(int x, int y, int z): x(x), y(y), z(z) {}
	Point3() =default;
  int x,y,z;
};

struct Parameters{
  Parameters(unsigned k, double xCubeL, double yCubeL, double zCubeL, int yCubeArr,
             int xCubeArr, int zCubeArr, int xArrGl=1, int yArrGl=1, int zArrGl=1):
    k(k), xCubeL(xCubeL), yCubeL(yCubeL), zCubeL(zCubeL), xCubeArr(xCubeArr), yCubeArr(yCubeArr), 
    zCubeArr(zCubeArr), xArrGl(xArrGl),yArrGl(yArrGl),zArrGl(zArrGl), pageSize(yCubeArr*xCubeArr) {}
  const unsigned k;                           //!< Number of neighbors to return
  const double xCubeL, yCubeL, zCubeL;        //!< Cube length in each dimension
  const int xCubeArr, yCubeArr, zCubeArr;     //!< Number of Cubes in CubeArray in each dimension
  const int xArrGl,yArrGl,zArrGl;                      //!< Number of CubeArrays in entire space
  const int pageSize;                         //!< Size of an "page" of all the columns and rows in CubeArray
};
/*
class MPIhandler{
public:
	//! Takes &argc, &argv
	MPIhandler(int* argc, char*** argv){
		error= MPI_Init(argc, argv);
		if (error) printf("[MPI]: MPI_init ERROR=%d\n", error);
		//TODO: define custom data type Point3
	}
	~MPIhandler(){ MPI_Finalize(); }

private:
	int error;
};
*/
