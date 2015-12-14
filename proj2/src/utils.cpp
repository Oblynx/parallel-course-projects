#include "utils.h"
#include <cmath>
using namespace std;

Cube& CubeArray::locateQ(const Element& q){
  const unsigned x= floor(q.x/param.xsize), y= floor(q.y/param.ysize),
                 z= floor(q.z/param.zsize);
  Cube& qCube= data_[x+param.cols*y+param.pageSize*z];
  return qCube;
}

std::deque<Element*> Search::query(const Element& q){
  init(cubeArray_.locateQ(q));
  search(q);
  while( nn_.size() < param.k ){
    expand();
    search(q);
  }
  return nn_.get_container();
}
void Search::search(const Element& q){
  for(auto&& cube : searchSpace_)
    for(Element& elt : cube->data_)
      if (nn_.size() < param.k) nn_.push(&elt);
      else if (nn_.top()->d(q) > elt.d(q)){
        nn_.top()->resetD();
        nn_.pop();
        nn_.push(&elt);
      } else elt.resetD();
}

void Search::init(Cube& center){
  searchLim_[0]= center.x-1, searchLim_[1]= center.x+1, searchLim_[2]= center.y-1;
  searchLim_[3]= center.y+1, searchLim_[4]= center.z-1, searchLim_[5]= center.z+1;
  for(int x= center.x-1; x<= center.x+1; x++)
    for(int y= center.y-1; y<= center.y+1; y++)
      for(int z= center.z-1; z<= center.z+1; z++){
        add({x,y,z});
        //searchSpace_.push_back(&cubeArray_[{x,y,z}]);
        //TOCHECK
      }
}
//! Calculate address for each new cube and retrieve its reference
//! 3 cases: either the cube exists in CubeArray or another processor has in (very unlikely) or out-of-bounds
// 
void Search::expand(){
  //TODO
  
}

void Search::add(Point3 cd){
  
}

//TODO: create Element point cloud...
//TODO: local->global coordinate transform
