#include <memory>
#include <vector>
#include <queue>
#include <deque>

//! Allows access to the underlying container in STL adapters like priority_queue
template <class Container>
class ContainerAccessor : public Container {
public:
    typedef typename Container::container_type container_type;
    container_type get_container() { return this->c; }
};

struct Point3{
  Point3(int x, int y, int z): x(x), y(y), z(z) {}
	Point3() =default;
  int x,y,z;
};

struct Parameters{
  Parameters(unsigned k, unsigned xsize, unsigned ysize, unsigned zsize, unsigned rows,
             unsigned cols, unsigned pages):
    k(k), xsize(xsize), ysize(ysize), zsize(zsize), rows(rows), cols(cols),
    pages(pages), pageSize(rows*cols) {}
  const unsigned k;    //!< Number of neighbors to return
  const unsigned xsize, ysize, zsize;   //!< Number of Elements in Cube in each dimension
  const unsigned rows, cols, pages;     //!< Number of Cubes in CubeArray in each dimension
  const unsigned pageSize;
};
struct Element{
  Element(double x, double y, double z): x(x), y(y), z(z) {}
  //! Calculate dist or return it, if it was previously calculated 
  double d(const Element& q){
    if (!distInit_) dist_= (q.x[0]-x[0])*(q.x[0]-x[0]) + (q.x[1]-x[1])*(q.x[1]-x[1]) + 
                          (q.x[2]-x[2])*(q.x[2]-x[2]);
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
  Cube(int x, int y, int z): x(x), y(y), z(z) {}
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
  //! Start a new search and return Q's nearest neighbors. The Elements' distance is *not* reset.
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
