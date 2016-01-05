#pragma once
#include "mpi_handler.h" 

//! C compatible struct with point and address of containing CubeArray(s)
struct PointAddress{
  Point3f p;
  int address[8];
  short addrUsed;
};

class All2allTransfer{
 public:
  //! Returns send request code to receive asynchronously
  template<typename F>
  All2allTransfer(F generator, const Parameters& param, MPIhandler& mpi, const int pointN,
                  const int procN): sSizeBuf(new int[procN]), rSizeBuf(new int[procN]), mpi(mpi),asyncRequest(mpi) {
    //PRINTF("\t###- Cube length: %f,%f,%f; CubeArray length: %d,%d,%d", param.xCubeL,param.yCubeL,param.zCubeL,param.xCubeArr,param.yCubeArr,param.zCubeArr);
    std::unique_ptr<PointAddress[]> buf(new PointAddress[pointN]);
    for(int i=0;i<procN;i++) sSizeBuf[i]=0, rSizeBuf[i]=0;
    //Generate the points in buffer
    for(int i=0;i<pointN;i++){
      buf[i]= generator(param);
      //Increase the size of every destination address
      for(int j=0; j<buf[i].addrUsed;j++){
        //PRINTF("[transfer#%d]: --> point and address: (%f,%f,%f);%d;%d\n",mpi.rank(),buf[i].p.x,buf[i].p.y,buf[i].p.z,buf[i].address[0],buf[i].addrUsed);
        sSizeBuf[buf[i].address[j]]++;
      }
    }
    /*{std::string showSizes;
      for(int i=0; i<procN; i++) showSizes+= std::to_string(sSizeBuf[i]), showSizes+= ';';
    PRINTF("[transfer#%d]: Send sizes = %s -|\n",mpi.rank(),showSizes.c_str());}*/
    COUT<<"\n[transfer#"<<mpi.rank()<<"]: Requesting Size communications\n";
    //All2all comms for size -- CAUTION! count is per process!!!
    asyncRequest.Ialltoall(sSizeBuf.get(),1,MPI_INT,rSizeBuf.get(),1);

    sdispl.reset(new int[procN]); //Start displacements for each proc in sendBuf
    sdispl[0]=0;
    for(int i=1; i<procN; i++) sdispl[i]= sdispl[i-1]+sSizeBuf[i-1];
    
    //COUT<<"\n[transfer#"<<mpi.rank()<<"]: Creating send buffer\n";
    //Create send buffer
    sendBuf.reset(new Point3f[sdispl[procN-1]+sSizeBuf[procN-1]]);
    for(int i=0;i<pointN;i++) for(int j=0; j<buf[i].addrUsed; j++)
        sendBuf[sdispl[buf[i].address[j]]++]= buf[i].p;
    
    //Wait for size comms to complete
    //COUT<<"[transfer#"<<mpi.rank()<<"]: Waiting for size comm\n";
    asyncRequest.wait();
    mpi.barrier();
    COUT<<"[transfer#"<<mpi.rank()<<"]: Size comm complete!\n";
    /*{std::string showSizes;
      for(int i=0; i<procN; i++) showSizes+= std::to_string(rSizeBuf[i]), showSizes+= ';';
    PRINTF("[transfer#%d]: Receive sizes = %s -|\n",mpi.rank(),showSizes.c_str());}*/

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
    COUT<<"[tranfer#"<<mpi.rank()<<"]: rcvSize_="<<rcvSize_<<'\n';
    rcvBuf.reset(new Point3f[rcvSize_]);
    COUT<<"[transfer#"<<mpi.rank()<<"]: Requesting point transfer comms\n";
    asyncRequest.Ialltoallv(sendBuf.get(),sSizeBuf.get(),sdispl.get(),mpi.typePoint3f(),
                            rcvBuf.get(),rSizeBuf.get(),rdispl.get());
  }
  std::unique_ptr<Point3f[]> get(int& rcvN) {
    //PRINTF("[getTransfer#%d]: enter\n",mpi.rank());
    asyncRequest.wait();
    sSizeBuf.reset(nullptr),rSizeBuf.reset(nullptr),sendBuf.reset(nullptr);
    sdispl.reset(nullptr), rdispl.reset(nullptr);
    rcvN= rcvSize_;
    return std::move(rcvBuf);
  }
 private:
  std::unique_ptr<int[]> sSizeBuf,rSizeBuf, sdispl,rdispl; //!<Size and displacement
  std::unique_ptr<Point3f[]> sendBuf,rcvBuf;
  int rcvSize_;
  MPIhandler& mpi;
  MPIhandler::AsyncRequest asyncRequest;
};

