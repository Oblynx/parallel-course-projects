#pragma once
#include <memory>
#include <functional>
#include <vector>
#include <queue>
#include <deque>
#include <future>
#include <cmath>
#include "utils.h"
#include "mpi_handler.h"

struct Element{
  Element(Point3f cd): x(cd.x), y(cd.y), z(cd.z) {}
  Element(float x, float y, float z): x(x), y(y), z(z) {}
  //! Calculate dist or return it, if it was previously calculated. The distance is memorized to allow for
  //  ordering the Elements in the priority queue 
  float distStateful(const Element& q){
    if (!distInit_) dist_= (q.x-x)*(q.x-x) + (q.y-y)*(q.y-y) + (q.z-z)*(q.z-z), distInit_= true;
    return dist_;
  }
  float dist(const Element& q){ return (q.x-x)*(q.x-x) + (q.y-y)*(q.y-y) + (q.z-z)*(q.z-z); }
  void resetD() { distInit_= false; }
  //! Used in priority queue. Must be called after dist has been initialized!
  bool operator<(const Element& other) const { return dist_ < other.dist_; }
  const float x,y,z;    //!< Position vector
private:
  float dist_;   //!< Memorize distance from current query
  bool distInit_= false;
};

//! The smallest indivisible part of the search space. Collection of all the Elements that occupy
//! a rectangular portion of the total space
struct Cube{
  Cube(const Parameters& param, int x, int y, int z): x(x), y(y), z(z), param(param) {/*TODO: reserve!*/}
  Cube& place(Point3f elt) { 
    //PRINTF("%f,%f,%f <- %d,%d,%d\n", elt.x,elt.y,elt.z, x,y,z);
    data_.push_back(elt); return *this; }
  const int x,y,z;    //!< Its coordinates in the array it belongs to (address= (x+overlap)+(y+overlap)*xCubeArr+(z+overlap)*yCubeArr*xCubeArr)
  const Parameters& param;
  std::vector<Element> data_;
};

//! Collection of all the boxes that a process accesses directly. (each process constructs 1)
//! Indexed in row-col-page order (x+xCubeArr*y+pageSize*z)
class CubeArray{
public:
  CubeArray(const Parameters& param, int xGl, int yGl, int zGl): param(param),x(xGl),y(yGl),z(zGl),
            //bndL({xGl*param.xCubeL*param.xCubeArr, yGl*param.yCubeL*param.yCubeArr,zGl*param.zCubeL*param.zCubeArr}),
            //bndH({(xGl+1)*param.xCubeL*param.xCubeArr,(yGl+1)*param.yCubeL*param.yCubeArr,(zGl+1)*param.zCubeL*param.zCubeArr}),
            locBndL({0-param.overlapCubes,0-param.overlapCubes,0-param.overlapCubes}),
            locBndH({param.xCubeArr+param.overlapCubes,param.yCubeArr+param.overlapCubes,param.zCubeArr+param.overlapCubes})
    {
    data_.reserve((param.xCubeArr+2*param.overlapCubes)*(param.yCubeArr+2*param.overlapCubes)*
                  (param.zCubeArr+2*param.overlapCubes));
    int x,y,z;
    for(z=0-param.overlapCubes; z<param.zCubeArr+param.overlapCubes; z++)
      for(y=0-param.overlapCubes; y<param.yCubeArr+param.overlapCubes; y++)
        for(x=0-param.overlapCubes; x<param.xCubeArr+param.overlapCubes; x++)
          data_.emplace_back(param,x,y,z);
    }
  //! Which box does Q belong to?
  Cube& locateQ(const Element& q);
  //! Return Cube at given (local) coordinates
  Cube& operator[](Point3 cd){
    const int overlap= param.overlapCubes;
    return data_[(cd.x+overlap) + (cd.y+overlap)*(param.xCubeArr+2*overlap) +
                 (cd.z+overlap)*(param.yCubeArr+2*overlap)*(param.xCubeArr+2*overlap)];
  }
  //! Place a new element in its corresponding Cube in CubeArray
  Cube& place(Point3f elt); 
  //! Calculate the distance of q from the boundary of given cube
  float distFromBoundary(Element q, Cube& cube);
  //! Global coordinates of point cd
  Point3 global(Point3 cd) {
    return {cd.x+param.xCubeArr*x, cd.y+param.yCubeArr*y, cd.z+param.zCubeArr*z};
  }
  Point3 local(Point3 cd) {
    return {cd.x-param.xCubeArr*x, cd.y-param.yCubeArr*y, cd.z-param.zCubeArr*z};
  }
private:
  std::vector<Cube> data_;
  const Parameters& param;
  const int x,y,z;  //!< CubeArray coordinates in global space
  //Point3f bndL,bndH;  //!< Lower and upper bounds
public:
  Point3 locBndL,locBndH;
};

//! Performs operations on the total search space, which is a CubeArray instance. Doesn't hold
//! any actual data, only manages references
class Search{
  typedef std::priority_queue<Element*, std::deque<Element*>, lessPtr<Element*>> EltMaxQ;
public:
  Search(CubeArray& cubeArray, const Parameters& param, MPIhandler& mpi): cubeArray_(cubeArray),
      param(param), mpi(mpi) {}
  //! Start a new search and return Q's nearest neighbors. The Elements' distance is *not* reset for Elements in the returned deque.
  std::deque<Element> query(const Element& q);
private:
  //! Search for new nearest neighbors in the constructed search space and expand the NN list
  void search(const Element& q, EltMaxQ& nn, std::deque<Cube*>& searchSpace);
  //! Add the next layer of boxes in 3D space to the search space and remove the old
  void expand(std::deque<Cube*>& searchSpace);
  //! Fetch a new Cube ref given the coordinates and add it to the search space
  //! 3 cases: 1. Cube exists in this process's CubeArray  2. Owned by another process  3. out-of-bounds
  void add(const Point3 coord, std::deque<Cube*>& searchSpace);
  //! Request Cube globCd from remote process processCd [MPI]
  void request(const int rank, const Point3 globCd);
  //! Wait for any MPI request that might have been initiated from expand to finish [future]
  void waitRequestsFinish();
  
  std::deque<MPIhandler::AsyncRequest> cubeRequests_;
	struct { Point3 l,h; } searchLim_;
  CubeArray& cubeArray_;
  const Parameters& param;
  MPIhandler& mpi;
};
