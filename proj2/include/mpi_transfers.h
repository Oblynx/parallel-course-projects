#pragma once
#include <memory>
#include "mpi_handler.h" 
#include <cassert>

//! C-compatible struct with point and address of containing CubeArray(s)
struct PointAddress{
  Point3f p;
  int address[8];
  short addrUsed;
};

//! Implements nonblocking all2all communications to exchange points and queries
class All2allTransfer{
 public:
  template<typename F>
  All2allTransfer(F generator, const Parameters& param, MPIhandler& mpi, const int pointN,
                  const int procN): sSizeBuf(new int[procN]), rSizeBuf(new int[procN]), pointN(pointN),
                  procN(procN), buf(new PointAddress[pointN]), mpi(mpi),asyncRequest(mpi) {
    for(int i=0;i<procN;i++) sSizeBuf[i]=0, rSizeBuf[i]=0;
    if(mpi.disabled) rcvSize_= pointN, rcvBuf.reset(new Point3f[pointN]);
    //Generate the points in buffer
    for(int i=0;i<pointN;i++){
      buf[i]= generator(param);
      //Increase the size of every destination address
      for(int j=0; j<buf[i].addrUsed;j++){
        assert(buf[i].address[j] < procN);
        sSizeBuf[buf[i].address[j]]++;
      }
      if(mpi.disabled) rcvBuf[i]= buf[i].p;
    }
  }
  void transfer(){
    if(mpi.disabled) return;
    /*{std::string showSizes;
      for(int i=0; i<procN; i++) showSizes+= std::to_string(sSizeBuf[i]), showSizes+= ';';
    PRINTF("[transfer#%d]: Send sizes = %s -|\n",mpi.rank(),showSizes.c_str());}*/
    //Everyone must have reached the last nonblocking collective communication before going to the next
    mpi.barrier();
    PRINTF("[transfer#%d]: Requesting Size comms\n",mpi.rank());
    //All2all comms for size -- CAUTION! count is per process!!!
    asyncRequest.Ialltoall(sSizeBuf.get(),1,MPI_INT,rSizeBuf.get(),1);

    sdispl.reset(new int[procN]); //Start displacements for each proc in sendBuf
    sdispl[0]=0;
    for(int i=1; i<procN; i++) sdispl[i]= sdispl[i-1]+sSizeBuf[i-1];
    
    //Create send buffer
    int totalSendP= sdispl[procN-1]+sSizeBuf[procN-1];
    sendBuf.reset(new Point3f[totalSendP]);
    for(int i=0;i<pointN;i++) for(int j=0; j<buf[i].addrUsed; j++){
        assert(sdispl[buf[i].address[j]] < totalSendP);
        sendBuf[sdispl[buf[i].address[j]]++]= buf[i].p;
    }
    PRINTF("[transfer#%d]: Waiting for size comms\n",mpi.rank());
    //Wait for size comms to complete
    asyncRequest.wait();
    mpi.barrier();
    PRINTF("[transfer#%d]: Size comm complete!\n",mpi.rank());
    /*{std::string showSizes;
      for(int i=0; i<procN; i++) showSizes+= std::to_string(rSizeBuf[i]), showSizes+= ';';
    PRINTF("[transfer#%d]: Receive sizes = %s -|\n",mpi.rank(),showSizes.c_str());}*/

    //size OK, communicate sendBuf
    sdispl[0]=0;
    for(int i=1; i<procN; i++) sdispl[i]= sdispl[i-1]+sSizeBuf[i-1];
    rdispl.reset(new int[procN]);
    rdispl[0]=0;
    for(int i=1; i<procN; i++) rdispl[i]= rdispl[i-1]+rSizeBuf[i-1];
    //Calculate the number of points this proc will receive
    rcvSize_=0;
    for(int i=0;i<procN;i++) rcvSize_+= rSizeBuf[i];
    PRINTF("[transfer#%d]: rcvSize=%d\n",mpi.rank(),rcvSize_);
    rcvBuf.reset(new Point3f[rcvSize_]);
    PRINTF("[transfer#%d]: Requesting point transfer comms\n",mpi.rank());
    asyncRequest.Ialltoallv(sendBuf.get(),sSizeBuf.get(),sdispl.get(),mpi.typePoint3f(),
                            rcvBuf.get(),rSizeBuf.get(),rdispl.get());
    buf.reset(nullptr);
  }
  std::unique_ptr<Point3f[]> get(int& rcvN) {
    asyncRequest.wait();
    sSizeBuf.reset(nullptr),rSizeBuf.reset(nullptr),sendBuf.reset(nullptr);
    sdispl.reset(nullptr), rdispl.reset(nullptr);
    rcvN= rcvSize_;
    return std::move(rcvBuf);
  }
 private:
  std::unique_ptr<int[]> sSizeBuf,rSizeBuf, sdispl,rdispl; //!<Size and displacement
  std::unique_ptr<Point3f[]> sendBuf,rcvBuf;
  const int pointN, procN;
  std::unique_ptr<PointAddress[]> buf;
  int rcvSize_;
  MPIhandler& mpi;
  MPIhandler::AsyncRequest asyncRequest;
};

