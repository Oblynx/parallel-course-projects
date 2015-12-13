#include <memory>
#include <vector>
#include <queue>
#include <deque>

struct Element{
  //! Calculate dist or return it, if it was previously calculated 
  double d(const Element& q){
    if (!distInit_) dist_= (q.x[0]-x[0])*(q.x[0]-x[0]) + (q.x[1]-x[1])*(q.x[1]-x[1]) + 
                          (q.x[2]-x[2])*(q.x[2]-x[2]);
    return dist_;
  }
  void resetD() { distInit_= false; }
  //! Used in priority queue. Must be called after dist has been initialized!
  bool operator<(const Element& other){ return dist_ < other.dist_; }
  double x[3];    //!< Position vector
private:
  double dist_;   //!< Memorize distance from current query
  bool distInit_= false;
};

//! Collection of all the Elements that occupy a rectangular portion of the total space
struct Box{
  double xlim[2], ylim[2], zlim[2];
  int address;  //!< Its address in the array it belongs to
  std::vector<Element> data_;
};

//! Collection of all the boxes that a process accesses directly.
//! Indexed in row-col-page order (x+rowSize*y+pageSize*z)
class BoxArray{
public:
  //! Which box does Q belong to?
  Box& locateQ(Element q);
private:
  std::vector<Box> data_;
};

class Search{
  typedef std::priority_queue<Element&, std::deque<Element&>> EltMaxHeap;
public:
  //! Start a new search and return Q's nearest neighbors
  std::deque<Element> query(Element& q);
  //! Search for new nearest neighbors in the constructed search space and expand the NN list
  void search();
  //! Add the next layer of boxes in 3D space to the search space and remove the old
  void expand();
  //! First expansion is special, because it keeps the central box containing the Query
  void init();
private:
  void add(Box& newBox) {}
  std::deque<Box&> searchSpace_;
  EltMaxHeap nn;
};

