#include <limits.h>
#include <stdexcept>
#include <iostream>
#include "mpi_handler.h"
using namespace std;

MPIhandler::MPIhandler(bool enable, int* argc, char*** argv): disabled(!enable) {
  if(enable){
    error= MPI_Init(argc, argv);  errorHandler();
    error= MPI_Comm_size(MPI_COMM_WORLD, &procN_);  errorHandler();
    error= MPI_Comm_rank(MPI_COMM_WORLD, &rank_);   errorHandler();
    // https://www.rc.colorado.edu/sites/default/files/Datatypes.pdf
    error= MPI_Type_contiguous(3, MPI_INT, &pT);    errorHandler();
    error= MPI_Type_commit(&pT);  errorHandler();
    error= MPI_Type_contiguous(3, MPI_FLOAT, &pfT);  errorHandler();
    error= MPI_Type_commit(&pfT);   errorHandler();
  }
}
MPIhandler::~MPIhandler() { if(!disabled) MPI_Finalize(); }
void MPIhandler::errorHandler() {
  if(error != MPI_SUCCESS){
    COUT <<"[MPIhandler]: Error!\n";
    char error_string[BUFSIZ];
    int errStrL, error_class;

    MPI_Error_class(error, &error_class);
    MPI_Error_string(error_class, error_string, &errStrL);
    fprintf(stderr, "\t!!! ERROR #%3d: %s !!!\n", rank_, error_string);
    MPI_Error_string(error, error_string, &errStrL);
    fprintf(stderr, "\t!!! ERROR #%3d: %s !!!\n", rank_, error_string);
  }
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
  mpi.errorHandler();
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
  mpi.errorHandler();
}
void MPIhandler::AsyncRequest::wait(){
  if(mpi.disabled) return;
  if(!requestInTransit) throw new runtime_error("[MPI_AsyncRequest]: Called wait while no request pending!\n");
  mpi.error= MPI_Wait(&pendingRequest, MPI_STATUS_IGNORE);  mpi.errorHandler();
  requestInTransit=false;
}
