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
  Parameters(unsigned k, double xsize, double ysize, double zsize, unsigned rows,
             unsigned cols, unsigned pages):
    k(k), xsize(xsize), ysize(ysize), zsize(zsize), rows(rows), cols(cols),
    pages(pages), pageSize(rows*cols) {}
  const unsigned k;    //!< Number of neighbors to return
  const double xsize, ysize, zsize;   //!< Cube length in each dimension
  const unsigned rows, cols, pages;     //!< Number of Cubes in CubeArray in each dimension
  const unsigned pageSize;
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
