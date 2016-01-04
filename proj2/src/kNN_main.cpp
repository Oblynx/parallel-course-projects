#include <math.h>
#include <iostream>
#include "kNNAlgo.h"
#include "mpi_handler.h"
using namespace std;

//! C compatible struct with point and address of containing CubeArray(s)
struct PointAddress{
  Point3f p;
  int address[8];
  short addrUsed;
};

//! Generates random points + corresponding *CubeArray* (aka proc) address
PointAddress pointGenerator(const Parameters& param){
  //PointAddress p{{xor128(),xor128(),xor128()},{0},0};
  PointAddress p{{(float)rand()/RAND_MAX,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX},{0},0};
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
  //PointAddress p {{xor128(),xor128(),xor128()},{0},1};
  PointAddress p{{(float)rand()/RAND_MAX,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX},{0},1};
  p.address[0]= (int)p.p.x/(param.xCubeL*param.xCubeArr)+
                (int)p.p.y/(param.yCubeL*param.yCubeArr)*param.xArrGl+
                (int)p.p.z/(param.zCubeL*param.zCubeArr)*param.yArrGl*param.xArrGl;
  return p;
}

//TODO: Parameter::overlap -> factor between [0,1] that is compared with the coord's fractional part

class All2allTransfer{
 public:
  //! Returns send request code to receive asynchronously
  template<typename F>
  All2allTransfer(F generator, const Parameters& param, MPIhandler& mpi, const int pointN,
                  const int procN): sSizeBuf(new int[procN]), rSizeBuf(new int[procN]), mpi(mpi),asyncRequest(mpi) {
    COUT<<"[transfer]: begin constructor\n";
    unique_ptr<PointAddress[]> buf(new PointAddress[pointN]);
    for(int i=0;i<procN;i++) sSizeBuf[i]=0, rSizeBuf[i]=0;
    //Generate the points in buffer
    for(int i=0;i<pointN;i++){
      buf[i]= generator(param);
      //Increase the size of every destination address
      for(int j=0; j<buf[i].addrUsed;j++){
        //COUT<<"[transfer]: point and address: "<<buf[i].p.x<<' '<<buf[i].p.y<<' '<<buf[i].p.z<<';'<<buf[i].address[0]<<';'<<buf[i].addrUsed<<'\n';
        sSizeBuf[buf[i].address[j]]++;
      }
    }
    COUT<<"[transfer]: Requesting MPI communications\n";
    //All2all comms for size
    asyncRequest.Ialltoall(sSizeBuf.get(),procN,MPI_INT,rSizeBuf.get(),procN);
    sdispl.reset(new int[procN]); //Start displacements for each proc in sendBuf
    sdispl[0]=0;
    for(int i=1; i<procN; i++){
      sdispl[i]= sdispl[i-1]+sSizeBuf[i-1];
      cout<<sdispl[i]<<' ';
    }
    
    COUT<<"\n[transfer]: Creating send buffer\n";
    //Create send buffer
    sendBuf.reset(new Point3f[sdispl[procN-1]+sSizeBuf[procN-1]]);
    for(int i=0;i<pointN;i++) for(int j=0; j<buf[i].addrUsed; j++){
      //cout<<sdispl[buf[i].address[j]]<<'\n';
      sendBuf[sdispl[buf[i].address[j]]++]= buf[i].p;
    }
    COUT<<"[transfer]: Waiting for size comm\n";
    //Wait for size comms to complete
    asyncRequest.wait();
    //size OK, communicate sendBuf
    sdispl[0]=0;
    for(int i=1; i<procN; i++) sdispl[i]= sdispl[i-1]+sSizeBuf[i-1];
    rdispl.reset(new int[procN]);
    rdispl[0]=0;
    for(int i=1; i<procN; i++) rdispl[i]= rdispl[i-1]+rSizeBuf[i-1];
    //TODO: Test rdispl
    //Calculate the number of points this proc will receive
    rcvSize_=0;
    for(int i=0;i<procN;i++) rcvSize_+= rSizeBuf[i];
    rcvBuf.reset(new Point3f[rcvSize_]);
    COUT<<"[transfer]: Requesting point transfer comms\n";
    asyncRequest.Ialltoallv(sendBuf.get(),sSizeBuf.get(),sdispl.get(),mpi.typePoint3f(),
                            rcvBuf.get(),rSizeBuf.get(),rdispl.get());
    COUT<<"[transfer]: Requested\n";
  }
  unique_ptr<Point3f[]> get() {
    asyncRequest.wait();
    sSizeBuf.reset(nullptr),rSizeBuf.reset(nullptr),sendBuf.reset(nullptr);
    sdispl.reset(nullptr), rdispl.reset(nullptr);
    return std::move(rcvBuf);
  }
  int pointsReceived() { return rcvSize_; }
 private:
  unique_ptr<int[]> sSizeBuf,rSizeBuf, sdispl,rdispl; //!<Size and displacement
  unique_ptr<Point3f[]> sendBuf,rcvBuf;
  int rcvSize_;
  MPIhandler& mpi;
  MPIhandler::AsyncRequest asyncRequest;
};

int main(int argc, char** argv){
  MPIhandler mpi(&argc, &argv);
  COUT << "MPI handler constructed\n";
  const int N=1<<20, Q=1<<17, P= mpi.procN(), rank= mpi.rank();
  //TODO: {x,y,z}ArrGl as function of P? (or simply input?)
  Parameters param(5,0,1, 0.1,0.1,0.1, 10,10,10, 1,1,1);
  //Generate N/P points
  All2allTransfer points(pointGenerator,param,mpi,N/P,P);
  COUT << "Points comm started\n";
  //Generate Q/P queries
  All2allTransfer queries(queryGenerator,param,mpi,Q/P,P);
  
  //Sync points
  auto rcvPoints= points.get();
  CubeArray cubeArray(param,rank%param.xArrGl,rank/param.xArrGl,rank/(param.xArrGl*param.yArrGl));   
  for(int i=0; i<points.pointsReceived(); i++) cubeArray.place(rcvPoints[i]);
  COUT<<"All points received\n";
  //for(int i=0; i<points.pointsReceived(); i++) PRINTF("%f,%f,%f\n", rcvPoints[i].x, rcvPoints[i].y, rcvPoints[i].z);

  //Sync queries
  auto rcvQ= queries.get();
  COUT<<"All queries received\n";
  Search search(cubeArray, param, mpi);
  for(int i=0; i<queries.pointsReceived(); i++) search.query(rcvQ[i]);

  //Test
  COUT<<"Testing\n";
  Point3f testQ {0.5,0.5,0.5};
  auto qres= search.query(testQ);
  printf("NN for (%f, %f, %f):\n", testQ.x, testQ.y, testQ.z);
  for(auto&& elt : qres)
    printf("\t-> (%f,%f,%f): d= %e\n", elt->x,elt->y,elt->z,elt->d(testQ));
  printf("\n");
  return 0; 
}
