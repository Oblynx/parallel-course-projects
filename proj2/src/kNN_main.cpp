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
  //Divide coord by CubeArray length (=_CubeL*_CubeArr)
  //Integral part: CubeArray coordinate
  //Fractional pt: whether neighbors should be included
  float frx= modf(p.p.x/(param.xCubeL*param.xCubeArr),&cdxf),
        fry= modf(p.p.y/(param.yCubeL*param.yCubeArr),&cdyf),
        frz= modf(p.p.z/(param.zCubeL*param.zCubeArr),&cdzf);
  int cdx= (int)cdxf, cdy= (int)cdyf, cdz= (int)cdzf;
  //Address of containing CubeArray
  p.address[p.addrUsed++]= cdx+ cdy*param.xArrGl+ cdz*param.yArrGl*param.xArrGl;
  //Addresses of neighboring CubeArrays where the point belongs due to overlap
  if(frx > 1-param.xOverlap && cdx+1 < param.xArrGl) p.address[p.addrUsed++]= p.address[0]+1;
  if(fry > 1-param.yOverlap && cdy+1 < param.yArrGl) p.address[p.addrUsed++]= p.address[0]+param.xArrGl;
  if(frz > 1-param.zOverlap && cdz+1 < param.zArrGl) p.address[p.addrUsed++]= p.address[0]+param.xArrGl*param.yArrGl;
  if(frx < param.xOverlap && cdx-1 >= 0) p.address[p.addrUsed++]= p.address[0]-1;
  if(fry < param.yOverlap && cdy-1 >= 0) p.address[p.addrUsed++]= p.address[0]-param.xArrGl;
  if(frz < param.zOverlap && cdz-1 >= 0) p.address[p.addrUsed++]= p.address[0]-param.xArrGl*param.yArrGl;
  return p;
}
//! Generates random points and assigns them only to the CubeArray that contains them
PointAddress queryGenerator(const Parameters& param){
  PointAddress p {{(float)rand(),(float)rand(),(float)rand()},{0},1};
  p.address[0]= (int)p.p.x/(param.xCubeL*param.xCubeArr)+
                (int)p.p.y/(param.yCubeL*param.yCubeArr)*param.xArrGl+
                (int)p.p.z/(param.zCubeL*param.zCubeArr)*param.yArrGl*param.xArrGl;
  return p;
}

//TODO: Parameter::overlap -> factor between [0,1] that is compared with the coord's fractional part

int main(int argc, char** argv){
  MPIhandler mpi(&argc, &argv);
  //Generate N/P points
  //Generate N/P queries
}
