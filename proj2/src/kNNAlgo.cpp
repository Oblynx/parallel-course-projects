#include "kNNAlgo.h"
#include <cmath>
#include <iostream>
using namespace std;

Cube& CubeArray::locateQ(const Element& q){
  const unsigned x= floor(q.x/param.xCubeL), y= floor(q.y/param.yCubeL),
                 z= floor(q.z/param.zCubeL);
  return data_[x+param.xCubeArr*y+param.pageSize*z];
}
std::deque<Element*> Search::query(const Element& q){
  EltMaxQAdapter nn;
  searchSpace_.clear();
  Cube& qloc= cubeArray_.locateQ(q);
  searchSpace_.push_back(&qloc);
  searchLim_.l.x= qloc.x-1, searchLim_.h.x= qloc.x+1, searchLim_.l.y= qloc.y-1;
  searchLim_.h.y= qloc.y+1, searchLim_.l.z= qloc.z-1, searchLim_.h.z= qloc.z+1;
  search(q,nn);
  if(nn.top()->d(q) > qloc.distFromBoundary(q))
    while( nn.size() < param.k ){
      COUT<<"Not found! Expanding\n"; 
      char c; cin >> c;
      expand();
      search(q,nn);
    }
  return nn.get_container();
}
void Search::search(const Element& q, EltMaxQAdapter& nn){
  for(auto&& cube : searchSpace_)
    for(auto&& elt : cube->data_)
      if (nn.size() < param.k) {
        elt.d(q);       //Force calculation of distance
        nn.push(&elt);
      }
      else if (nn.top()->d(q) > elt.d(q)){
        PRINTF("Inserted: (%f,%f,%f)\tDist: %f, max: %f\n", elt.x,elt.y,elt.z,elt.d(q),nn.top()->d(q));
        nn.top()->resetD();
        nn.pop();
        nn.push(&elt);
      } else elt.resetD();
}
//! Calculate address for each new cube and retrieve its reference
//! 3 cases: either the cube exists in CubeArray or another processor has it (very unlikely) or out-of-bounds
void Search::expand(){
	// Produce possible coordinates
	int x,y,z;
	//Cube expansion faces:
		//1. xl-1, [yl-1,yh+1], [zl-1,zh+1]
		//2. xh+1,     ...    ,     ...    
	for(int i=0;i<2;i++){
		x= (i==0)? searchLim_.l.x-1: searchLim_.h.x+1;
		for(y=searchLim_.l.y-1; y<=searchLim_.h.y+1; y++)
			for(z=searchLim_.l.z-1; z<=searchLim_.h.z+1; z++)
				add({x,y,z});
	}
		//3. [xl,xh], yl-1, [zl-1,zh+1]
		//4.   ...  , yh+1,     ...    
	for(int i=0; i<2; i++){
		y= (i==0)? searchLim_.l.y-1: searchLim_.h.y+1;
		for(x= searchLim_.l.x; x<=searchLim_.h.x; x++)
			for(z=searchLim_.l.z-1; z<=searchLim_.h.z+1; z++)
				add({x,y,z});
	}
		//5.  ...  , [yl,yh], zl-1
		//6.  ...  ,   ...  , zh+1
	for(int i=0; i<2; i++){
		z= (i==0)? searchLim_.l.z-1: searchLim_.h.z+1;
		for(x= searchLim_.l.x; x<=searchLim_.h.x; x++)
			for(y=searchLim_.l.y; z<=searchLim_.h.y; z++)
				add({x,y,z});
	}
  searchLim_.l.x--, searchLim_.h.x++, searchLim_.l.y--;
  searchLim_.h.y++, searchLim_.l.z--, searchLim_.h.z++;
  waitRequestsFinish();
}

void Search::add(Point3 cd){
	char x,y,z;
	x= (cd.x<0)? 0: (cd.x<static_cast<int>(param.xCubeArr ))?  1:  2;
	y= (cd.y<0)? 0: (cd.y<static_cast<int>(param.yCubeArr ))?  4:  8;
	z= (cd.z<0)? 0: (cd.z<static_cast<int>(param.zCubeArr))? 16: 32;
  if((x|y|z) == 21) searchSpace_.push_back(&cubeArray_[cd]);	//local CubeArray
  else{
    auto glob= cubeArray_.global(cd);
    //If ! out-of-bounds, request cube from neighbor
    if (!(glob.x<0 || glob.y<0 || glob.z<0 || glob.x>param.xArrGl*param.xCubeArr ||
          glob.y>param.yArrGl*param.yCubeArr || glob.z>param.zArrGl*param.zCubeArr)){
      cout << "[expand]: ERROR! "<<" not in local CubeArray! "<<cd.x<<' '<<cd.y<<' '<<cd.z<<"\n";
      cout << "\tGlobal: "<<glob.x<<' '<<glob.y<<' '<<glob.z<<'\n';
      //The coordinates of the CubeArray that contains the point 
      auto containingCubeArrayCd= Point3(glob.x/param.xArrGl, glob.y/param.yArrGl, glob.z/param.zArrGl);
      request(containingCubeArrayCd, glob);
    }
  }
}

void Search::request(Point3 processCd, Point3 globCd){
  cubeRequests_.push_back(mpi.IsendCoordinates(globCd, 1, param.rank(processCd)));
}
void Search::waitRequestsFinish(){
}
//TODO: create Element point cloud...
