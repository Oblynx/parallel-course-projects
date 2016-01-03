#include <math.h>
#include "kNNAlgo.h"
#include "mpi_handler.h"
using namespace std;

//! C compatible struct with point and address of containing CubeArray(s)
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

//! Returns send request code to receive asynchronously
template<typename F>
int allComm(F generator, const Parameters& param, MPIhandler& mpi, const int pointN,
            const int procN, int* rcvBuf){
  unique_ptr<PointAddress[]> buf(new PointAddress[pointN]);
  unique_ptr<int[]> sSizeBuf(new int[procN]), rSizeBuf(new int[procN]);
  unique_ptr<Point3f[]> sendBuf;
  for(int i=0;i<procN;i++) sSizeBuf[i]=0, rSizeBuf[i]=0;
  //Generate the points in buffer
  for(int i=0;i<pointN;i++){
    buf[i]= generator(param);
    //Increase the size of every destination address
    for(int j=0; j<buf[i].addrUsed;j++) sSizeBuf[buf[i].address[j]]++;
  }
  //All2all comms for size
  int sSizeReq= mpi.Ialltoall(sSizeBuf.get(),procN,MPI_INT,rSizeBuf.get(),procN);
  unique_ptr<int[]> sAddrCumul(new int[procN]);   //Start displacements for each proc in sendBuf
  sAddrCumul[0]=0;
  for(int i=1; i<procN; i++) sAddrCumul[i]= sAddrCumul[i-1]+sSizeBuf[i-1];
  
  //Create send buffer
  for(int i=0;i<pointN;i++) for(int j=0; j<buf[i].addrUsed; j++)
    sendBuf[sAddrCumul[buf[i].address[j]]++]= buf[i].p;
  //Wait for size comms to complete
  mpi.wait(sSizeReq);
  //size OK, communicate sendBuf
  //TODO: Alltoallv send
  return mpi.Ialltoallv(sendBuf.get(),sSizeBuf.get(),procN,mpi.getPoint3f(),rcvBuf,rSizeBuf.get());
}

int main(int argc, char** argv){
  MPIhandler mpi(&argc, &argv);
  int N=1<<20, Q=1<<16, P=1;
  Parameters param(5,0,1, 0.1,0.1,0.1, 10,10,10);
  //Generate N/P points
  auto recvPoints= allComm(pointGenerator, param,mpi,N,P);
  //Generate N/P queries
  auto recvQueries= allComm(queryGenerator, param,mpi,N,P);
}
