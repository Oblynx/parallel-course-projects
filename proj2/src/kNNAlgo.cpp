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
  EltMaxQ nn;
  deque<Cube*> searchSpace;
  Cube& qloc= cubeArray_.locateQ(q);
  searchSpace.push_back(&qloc);
  searchLim_.l.x= qloc.x, searchLim_.h.x= qloc.x, searchLim_.l.y= qloc.y;
  searchLim_.h.y= qloc.y, searchLim_.l.z= qloc.z, searchLim_.h.z= qloc.z;
  search(q,nn,searchSpace);
  if(nn.empty() || nn.top()->d(q) > qloc.distFromBoundary(q))
    while( nn.size() < param.k ){
      COUT<<"Not found! Expanding\n"; 
      expand(searchSpace);
      COUT<<"In searchSpace:\n";
      for(auto&& cube : searchSpace) PRINTF("%d,%d,%d\n", cube->x,cube->y,cube->z);
      search(q,nn,searchSpace);
    }
  deque<Element*> results;
  while(!nn.empty()){
    results.push_front(nn.top());
    nn.pop();
  } 
  return results;
}
void Search::search(const Element& q, EltMaxQ& nn, deque<Cube*>& searchSpace){
  for(auto&& cube : searchSpace)
    for(auto&& elt : cube->data_)
      if (nn.size() < param.k) {  //This happens initially
        elt.d(q);       //Force calculation of distance
        nn.push(&elt);
      } else if (nn.top()->d(q) > elt.d(q)){  //If better candidate
        PRINTF("Inserted: (%f,%f,%f)\tDist: %f, max: %f\n", elt.x,elt.y,elt.z,elt.d(q),nn.top()->d(q));
        nn.top()->resetD();
        nn.pop();
        nn.push(&elt);
      } else elt.resetD();        //If not a better candidate
}
//! Calculate address for each new cube and retrieve its reference
//! 3 cases: either the cube exists in CubeArray or another processor has it (very unlikely) or out-of-bounds
void Search::expand(deque<Cube*>& searchSpace){
  searchSpace.clear();
	// Produce possible coordinates
	int x,y,z;
	//Cube expansion faces:
		//1. xl-1, [yl-1,yh+1], [zl-1,zh+1]
		//2. xh+1,     ...    ,     ...    
	for(int i=0;i<2;i++){
		x= (i==0)? searchLim_.l.x-1: searchLim_.h.x+1;
		for(y=searchLim_.l.y-1; y<=searchLim_.h.y+1; y++)
			for(z=searchLim_.l.z-1; z<=searchLim_.h.z+1; z++)
				add({x,y,z},searchSpace);
	}
		//3. [xl,xh], yl-1, [zl-1,zh+1]
		//4.   ...  , yh+1,     ...    
	for(int i=0; i<2; i++){
		y= (i==0)? searchLim_.l.y-1: searchLim_.h.y+1;
		for(x= searchLim_.l.x; x<=searchLim_.h.x; x++)
			for(z=searchLim_.l.z-1; z<=searchLim_.h.z+1; z++)
				add({x,y,z},searchSpace);
	}
		//5.  ...  , [yl,yh], zl-1
		//6.  ...  ,   ...  , zh+1
	for(int i=0; i<2; i++){
		z= (i==0)? searchLim_.l.z-1: searchLim_.h.z+1;
		for(x= searchLim_.l.x; x<=searchLim_.h.x; x++)
			for(y=searchLim_.l.y; y<=searchLim_.h.y; y++)
				add({x,y,z},searchSpace);
	}
  auto globL= cubeArray_.global(searchLim_.l), globH= cubeArray_.global(searchLim_.h);
  if(globL.x>0) searchLim_.l.x--;
  if(globL.y>0) searchLim_.l.y--;
  if(globL.z>0) searchLim_.l.z--;
  if(globH.x<param.xArrGl*param.xCubeArr-1) searchLim_.h.x++;
  if(globH.y<param.yArrGl*param.yCubeArr-1) searchLim_.h.y++;
  if(globH.z<param.zArrGl*param.zCubeArr-1) searchLim_.h.z++;
  waitRequestsFinish();
  //TODO: Add new cubes after requests have finished
}

void Search::add(Point3 cd, deque<Cube*>& searchSpace){
	char x,y,z;
	x= (cd.x<0)? 0: (cd.x<static_cast<int>(param.xCubeArr ))?  1:  2;
	y= (cd.y<0)? 0: (cd.y<static_cast<int>(param.yCubeArr ))?  4:  8;
	z= (cd.z<0)? 0: (cd.z<static_cast<int>(param.zCubeArr))? 16: 32;
  if((x|y|z) == 21) searchSpace.push_back(&cubeArray_[cd]);	//local CubeArray
  else{
    auto glob= cubeArray_.global(cd);
    //If ! out-of-bounds, request cube from neighbor
    if (!(glob.x<0 || glob.y<0 || glob.z<0 || glob.x>=param.xArrGl*param.xCubeArr ||
          glob.y>=param.yArrGl*param.yCubeArr || glob.z>=param.zArrGl*param.zCubeArr)){
      COUT << "[expand]: ERROR! "<<" not in local CubeArray! "<<cd.x<<','<<cd.y<<','<<cd.z<<"\n";
      COUT << "\tGlobal: "<<glob.x<<','<<glob.y<<','<<glob.z<<'\n';
      //The coordinates of the CubeArray that contains the point 
      Point3 containingCubeArrayCd {glob.x/param.xArrGl, glob.y/param.yArrGl, glob.z/param.zArrGl};
      request(containingCubeArrayCd, glob);
    }//else PRINTF("[expand]: Out-of-bounds cube: %d,%d,%d\n", cd.x,cd.y,cd.z);
  }
}

void Search::request(Point3 processCd, Point3 globCd){
  cubeRequests_.emplace_back(MPIhandler::AsyncRequest(mpi));
  cubeRequests_.back().IsendCoordinates(globCd, 1, param.rank(processCd));
}
void Search::waitRequestsFinish(){
}
//TODO: create Element point cloud...
