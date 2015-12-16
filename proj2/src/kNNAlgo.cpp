#include "kNNAlgo.h"
#include <cmath>
#include <iostream>
using namespace std;

Cube& CubeArray::locateQ(const Element& q){
  const unsigned x= floor(q.x/param.xsize), y= floor(q.y/param.ysize),
                 z= floor(q.z/param.zsize);
  return data_[x+param.cols*y+param.pageSize*z];
}
std::deque<Element*> Search::query(const Element& q){
  EltMaxQAdapter nn;
  searchSpace_.clear();
  Cube& qloc= cubeArray_.locateQ(q);
  init(qloc);
  //locate, sesp
  /*
  printf("Q located in: (%d,%d,%d)\n", qloc.x,qloc.y,qloc.z);
  for(auto&& elt : searchSpace_)
    printf("\t-> (%d,%d,%d)\n", elt->x,elt->y,elt->z);
  */
  search(q,nn);
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

void Search::init(Cube& center){
  searchLim_.l.x= center.x-1, searchLim_.h.x= center.x+1, searchLim_.l.y= center.y-1;
  searchLim_.h.y= center.y+1, searchLim_.l.z= center.z-1, searchLim_.h.z= center.z+1;
  for(int x= center.x-1; x<= center.x+1; x++)
    for(int y= center.y-1; y<= center.y+1; y++)
      for(int z= center.z-1; z<= center.z+1; z++)
        add({x,y,z});
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
  //TODO: New searchLim if out-of-bounds?
}

void Search::add(Point3 cd){
	char nb=0, x,y,z;		// 0-26
	x= (cd.x<0)? 0: (cd.x<static_cast<int>(param.cols ))?  1:  2;
	y= (cd.y<0)? 0: (cd.y<static_cast<int>(param.rows ))?  4:  8;
	z= (cd.z<0)? 0: (cd.z<static_cast<int>(param.pages))? 16: 32;
	switch (x|y|z) {
		case 0:		nb=0; break;
			//2 zeros
		case 1:		nb=1; break;
		case 2:		nb=2; break;
		case 4:		nb=3; break;
		case 8:		nb=4; break;
		case 16:	nb=5; break;
		case 32:	nb=6; break;

			//1 zero
		case 5:		nb=7; break;
		case 9:		nb=8; break;
		case 17:	nb=9; break;
		case 33:	nb=10; break;

		case 6:		nb=11; break;
		case 10:	nb=12; break;
		case 18:	nb=13; break;
		case 34:	nb=14; break;

		case 20:	nb=15; break;
		case 36:	nb=16; break;
		case 24:	nb=17; break;
		case 40:	nb=18; break;
			//0 zero
		case 21:	nb=19; break; 
		case 37:	nb=20; break;
		case 25:	nb=21; break;
		case 41:	nb=22; break;
		case 22:	nb=23; break;
		case 38:	nb=24; break;
		case 26:	nb=25; break;
		case 42:	nb=26; break;
	}
	if (nb == 19) searchSpace_.push_back(&cubeArray_[cd]);	//local CubeArray
	else{
    cout << "[expand]: ERROR! "<<nb<<" not in local CubeArray! "<<cd.x<<' '<<cd.y<<' '<<cd.z<<"\n";
		//TODO: Not in local CubeArray; needs global coord transform
	}
}

//TODO: create Element point cloud...
//TODO: local->global coordinate transform
