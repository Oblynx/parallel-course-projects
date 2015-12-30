#include <kNNAlgo.h>
#include <math.h>

struct PointAddress{
  Point3f p;
  int address[8];
  char addrUsed;
};

//! Generates random points + corresponding *CubeArray* (aka proc) address
PointAddress pointGenerator(const Parameters& param){
  PointAddress p{{(float)rand(),(float)rand(),(float)rand()},{0},0};
  char x,y,z;
  float cdxf,cdyf,cdzf; 
  float frx= modf(p.p.x/param.xCubeL, &cdxf), fry= modf(p.p.y/param.yCubeL, &cdyf),
        frz= modf(p.p.z/param.zCubeL, &cdzf);
  int cdx= (int)cdxf, cdy= (int)cdyf, cdz= (int)cdzf;
  p.address[p.addrUsed++]= cdx+ cdy*param.xCubeArr+ cdz*param.yCubeArr*param.xCubeArr; 
  if (frx > 1-param.xOverlap && cdx+1 < param.xArrGl*param.xCubeArr) p.address[p.addrUsed++]= p.address[0]+1;
  if (frx < param.xOverlap && cdx-1 > 0) p.address[p.addrUsed++]= p.address[0]+1;
  //if cdx> ???*x -> belongs to upper
  //if cdx< ???*x -> belongs to lower
  //Same for y,z
  //if x && y -> diag
  //every combination x-y,y-z,z-x and x-y-z
  return p;
}

//TODO: Parameter::overlap -> factor between [0,1] that is compared with the coord's fractional part

int main(int argc, char** argv){
  MPIhandler mpi(&argc, &argv);
  //Generate N/P points
  //Generate N/P queries
}
