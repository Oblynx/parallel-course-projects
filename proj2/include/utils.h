#pragma once
#include <cstdio>
#include <future>
#include <mpi.h>

#ifdef DEBUG
  #define PRINTF printf
  #define COUT cout
#else
  #define PRINTF while(0) printf
  #define COUT   while(0) cout
#endif

float xor128();
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

struct Point3f{
  Point3f(float x, float y, float z): x(x), y(y), z(z) {}
	Point3f() =default;
  float x,y,z;
};
struct Point3{
  Point3(int x, int y, int z): x(x), y(y), z(z) {}
	Point3() =default;
  int x,y,z;
};

struct Parameters{
  Parameters(unsigned k, float xCubeL, float yCubeL, float zCubeL, int yCubeArr,
             int xCubeArr, int zCubeArr, int xArrGl=1, int yArrGl=1, int zArrGl=1):
    k(k), xCubeL(xCubeL), yCubeL(yCubeL), zCubeL(zCubeL), xCubeArr(xCubeArr), yCubeArr(yCubeArr), 
    zCubeArr(zCubeArr), xArrGl(xArrGl),yArrGl(yArrGl),zArrGl(zArrGl), pageSize(yCubeArr*xCubeArr) {}
  const unsigned k;                           //!< Number of neighbors to return
  const float xCubeL, yCubeL, zCubeL;         //!< Cube length in each dimension
  const int xCubeArr, yCubeArr, zCubeArr;     //!< Number of Cubes in CubeArray in each dimension
  const int xArrGl,yArrGl,zArrGl;             //!< Number of CubeArrays in entire space
  const int pageSize;                         //!< Size of an "page" of all the columns and rows in CubeArray
  const int ranks[];

  int rank(Point3 pCd){
    return ranks[pCd.x+pCd.y*xArrGl+pCd.z*yArrGl*xArrGl];
  }
};

class MPIhandler{
public:
	//! Takes &argc, &argv
	MPIhandler(int* argc, char*** argv){
		error= MPI_Init(argc, argv);
		if (error) printf("[MPI]: MPI_init ERROR=%d\n", error);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Type_contiguous(3, MPI_INT, &pT);
    MPI_Type_commit(&pT);
    MPI_Type_contiguous(3, MPI_FLOAT, &pfT);
    MPI_Type_commit(&pfT);
		//TODO: define custom data type Point3
	}
	~MPIhandler(){ MPI_Finalize(); }
  std::future<void> IsendCoordinates(Point3 cd, int n, int dest);

private:
  typedef struct { float x,y,z;} c_Point3f;
  typedef struct { int    x,y,z;} c_Point3;
  MPI_Datatype pT, pfT;
	int error, rank_;
};

