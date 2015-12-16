#pragma once
#include <memory>
#include <vector>
#include <queue>
#include <deque>
#include "utils.h"

struct Element{
  Element(Point3 cd): x(cd.x), y(cd.y), z(cd.z) {}
  Element(double x, double y, double z): x(x), y(y), z(z) {}
  //! Calculate dist or return it, if it was previously calculated 
  double d(const Element& q){
    if (!distInit_) dist_= (q.x-x)*(q.x-x) + (q.y-y)*(q.y-y) + (q.z-z)*(q.z-z);
    return dist_;
  }
  void resetD() { distInit_= false; }
  //! Used in priority queue. Must be called after dist has been initialized!
  bool operator<(const Element& other){ return dist_ < other.dist_; }
  const double x,y,z;    //!< Position vector
private:
  double dist_;   //!< Memorize distance from current query
  bool distInit_= false;
};

//! The smallest indivisible part of the search space. Collection of all the Elements that occupy
//! a rectangular portion of the total space
struct Cube{
  Cube(int x, int y, int z): x(x), y(y), z(z) {/*TODO: reserve!*/}
  void place(Point3 elt) { data_.emplace_back(elt); }
  double xlim[2], ylim[2], zlim[2];
  const int x,y,z;    //!< Its coordinates in the array it belongs to (address= x+y*cols+z*pageSize)
  std::vector<Element> data_;
};

//! Collection of all the boxes that a process accesses directly. (each process constructs 1)
//! Indexed in row-col-page order (x+cols*y+pageSize*z)
class CubeArray{
public:
  CubeArray(const Parameters& param): param(param) {
    data_.reserve(param.rows*param.cols*param.pages);
    unsigned x,y,z;
    for(x=0; x<param.cols; x++) for(y=0; y<param.rows; y++) for(z=0; z<param.pages; z++)
      data_.emplace_back(x,y,z);
  }
  //! Which box does Q belong to?
  Cube& locateQ(const Element& q);
  //! Return element at given coordinates
  Cube& operator[](Point3 coord){
    return data_[coord.x+ coord.y*param.cols+ coord.z*param.pageSize];
  }
  void place(Point3 cd) { locateQ(cd).place(cd); }
private:
  std::vector<Cube> data_;
  const Parameters& param;
  //! Global neighbor IDs; 0 -> out-of-bounds. Order: -x,+x,-y,+y,-z,+z
  const unsigned neighborID[6]={0,0,0,0,0,0}; //TODO!  
};

//! Performs operations on the total search space, which is a CubeArray instance. Doesn't hold
//! any actual data, only manages references
class Search{
  typedef std::priority_queue<Element*, std::deque<Element*>> EltMaxQ;
  typedef ContainerAccessor<EltMaxQ> EltMaxQAdapter;
public:
  Search(CubeArray& cubeArray, const Parameters& param): cubeArray_(cubeArray), param(param) {}
  //! Start a new search and return Q's nearest neighbors. The Elements' distance is *not* reset for Elements in the returned deque.
  std::deque<Element*> query(const Element& q);
  //! Search for new nearest neighbors in the constructed search space and expand the NN list
  void search(const Element& q);
  //! Add the next layer of boxes in 3D space to the search space and remove the old
  void expand();
  //! First expansion is special, because it keeps the central box containing the Query
  void init(Cube& center);
private:
  //! Fetch a new Cube ref given the coordinates and add it to the search space
  //! 3 cases: 1. Cube exists in this process's CubeArray  2. Owned by another process  3. out-of-bounds
  void add(Point3 coord);
  std::deque<Cube*> searchSpace_;
	struct { Point3 l,h; } searchLim_;
  EltMaxQAdapter nn_;
  CubeArray& cubeArray_;
  const Parameters& param;
};
