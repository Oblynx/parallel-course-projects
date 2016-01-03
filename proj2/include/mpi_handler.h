#pragma once
#include <mpi.h>
#include <unordered_map>
#include "utils.h"

class MPIhandler{
public:
	//! Takes &argc, &argv
	MPIhandler(int* argc, char*** argv);
  MPIhandler(char): disabled(true) {}
	~MPIhandler();
  int IsendCoordinates(Point3 cd, int n, int dest);
  int Ialltoall(const void* sendBuf, int sendCnt, MPI_Datatype
                datatype, void* rcvBuf, int rcvCnt);
  int Ialltoallv(const void* sendBuf, const int sendCnt[], const int sCntN, MPI_Datatype
                 datatype, void* rcvBuf, const int rcvCnt[]);
  void wait(int request){
    MPI_Status status;
    MPI_Wait(&pendingComms.at(request), &status);
    //TODO: Check status!
  }
  MPI_Datatype getPoint3f(){ return pfT; }
  MPI_Datatype getPoint3() { return pT;  }
private:
  MPI_Datatype pT, pfT;
  std::unordered_map<int,MPI_Request> pendingComms;
	int error, rank_, serial;
  const char disabled=false;
};

