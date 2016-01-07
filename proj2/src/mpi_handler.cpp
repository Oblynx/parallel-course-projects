#include <limits.h>
#include <stdexcept>
#include <iostream>
#include "mpi_handler.h"
using namespace std;

MPIhandler::MPIhandler(bool enable, int* argc, char*** argv): disabled(!enable) {
  if(enable){
    error= MPI_Init(argc, argv);
    if (error) printf("[MPI]: MPI_init ERROR=%d\n", error);
    MPI_Comm_size(MPI_COMM_WORLD, &procN_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    // https://www.rc.colorado.edu/sites/default/files/Datatypes.pdf
    MPI_Type_contiguous(3, MPI_INT, &pT);
    MPI_Type_commit(&pT);
    MPI_Type_contiguous(3, MPI_FLOAT, &pfT);
    MPI_Type_commit(&pfT);
  }
}
MPIhandler::~MPIhandler() { if(!disabled) MPI_Finalize(); }
void MPIhandler::AsyncRequest::IsendCoordinates(Point3 cd, int n, int dest){
  //TODO
  throw new runtime_error("### Called AsyncRequest::IsendCoordinates!!! ###\n");
}
//! MPI_Ialltoall wrapper
void MPIhandler::AsyncRequest::Ialltoall(const void* sendBuf, const int sendCnt, MPI_Datatype type, void* rcvBuf, const int rcvCnt){
  if(mpi.disabled){
    rcvBuf= (void*)sendBuf;
    return;
  }
  if(requestInTransit) throw new runtime_error("[MPI_AsyncRequest]: Attempted to start transit\
                                                while another is pending.\n");
  requestInTransit=true;
  mpi.error= MPI_Ialltoall(sendBuf,sendCnt,type,rcvBuf,rcvCnt,type,MPI_COMM_WORLD,&pendingRequest);
  if(mpi.error) PRINTF("--> [MPI]: Error in Ialltoall comm!!! errcode=%d",mpi.error);
}
//! MPI Ialltoallv wrapper
void MPIhandler::AsyncRequest::Ialltoallv(const void* sendBuf, const int sendCnt[], const int sdispl[],
                                          MPI_Datatype type,void* rcvBuf, const int rcvCnt[], const int rdispl[]){
  if(mpi.disabled){
    rcvBuf= (void*)sendBuf;
    return;
  }
  if(requestInTransit) throw new runtime_error("[MPI_AsyncRequest]: Attempted to start transit\
                                                while another is pending.\n");
  requestInTransit=true;
  mpi.error= MPI_Ialltoallv(sendBuf,sendCnt,sdispl,type,rcvBuf,rcvCnt,rdispl,
                            type,MPI_COMM_WORLD,&pendingRequest);
}
void MPIhandler::AsyncRequest::wait(){
  if(mpi.disabled) return;
  if(!requestInTransit) throw new runtime_error("[MPI_AsyncRequest]: Called wait while no request pending!\n");
  MPI_Wait(&pendingRequest, MPI_STATUS_IGNORE);
  requestInTransit=false;
}
