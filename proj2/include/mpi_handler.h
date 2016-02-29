#pragma once
#include <mpi.h>
#include <unordered_map>
#include "utils.h"

class MPIhandler{
public:
	//! Takes &argc, &argv
	MPIhandler(bool enable, int* argc=NULL, char*** argv=NULL);
	~MPIhandler();
  void errorHandler();
  void barrier() {
    if(!disabled){
      error= MPI_Barrier(MPI_COMM_WORLD); errorHandler();
    }
  }
  MPI_Datatype typePoint3f(){ return pfT; }
  MPI_Datatype typePoint3() { return pT;  }
  int procN() { return (disabled)? 1: procN_; }
  int rank() { return (disabled)? 0: rank_; }
  class AsyncRequest{
   public:
    AsyncRequest(MPIhandler& mpi): mpi(mpi) {}
    void IsendCoordinates(Point3 cd, int n, int dest);
    void Ialltoall(const void* sendBuf, int sendCnt, MPI_Datatype
                  datatype, void* rcvBuf, int rcvCnt);
    void Ialltoallv(const void* sendBuf, const int sendCnt[], const int sdispl[], MPI_Datatype
                   datatype, void* rcvBuf, const int rcvCnt[], const int rdispl[]);
    void wait();
   private:
    bool requestInTransit=false;
    MPI_Request pendingRequest;
    MPIhandler& mpi;
  };
  const char disabled;
private:
  MPI_Datatype pT, pfT;
	int error, rank_, procN_;
};


