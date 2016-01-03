#include <limits.h>
#include <stdexcept>
#include "mpi_handler.h"
using namespace std;

MPIhandler::MPIhandler(int* argc, char*** argv): serial(0) {
  error= MPI_Init(argc, argv);
  if (error) printf("[MPI]: MPI_init ERROR=%d\n", error);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Type_contiguous(3, MPI_INT, &pT);
  MPI_Type_commit(&pT);
  MPI_Type_contiguous(3, MPI_FLOAT, &pfT);
  MPI_Type_commit(&pfT);
  //TODO: define custom data type Point3
}
MPIhandler::~MPIhandler() { if(!disabled) MPI_Finalize(); }
int MPIhandler::IsendCoordinates(Point3 cd, int n, int dest){
  //TODO
}

//! MPI_Ialltoall wrapper that returns a future to check completion
int MPIhandler::Ialltoall(const void* sendBuf, int sendCnt, MPI_Datatype type, void* rcvBuf, int rcvCnt){
  MPI_Status status;
  MPI_Request req;
  if(serial>=INT_MAX-1) throw new std::overflow_error("[MPIhandler]: Pending comms serial num overflow!\n");
  serial++;
  pendingComms.insert({serial,req});
  error= MPI_Ialltoall(sendBuf,sendCnt,type,rcvBuf,rcvCnt,type,MPI_COMM_WORLD,&pendingComms.at(serial));
  return serial;
}

int MPIhandler::Ialltoallv(const void* sendBuf, const int sendCnt[], const int sCntN, MPI_Datatype type,
                           void* rcvBuf, const int rcvCnt[]){
  MPI_Status status;
  MPI_Request req;
  if(serial>=INT_MAX-1) throw new std::overflow_error("[MPIhandler]: Pending comms serial num overflow!\n");
  serial++;
  pendingComms.insert({serial,req});
  unique_ptr<int[]> sdispl(new int[sCntN]), rdispl(new int[sCntN]);
  sdispl[0]=0;
  for(int i=1; i<sCntN; i++) sdispl[i]= sdispl[i-1]+sendCnt[i-1];
  //TODO: rdispl
  error= MPI_Ialltoallv(sendBuf,sendCnt,sdispl.get(),);
  //TODO: unique_ptr lifetimes too short!!! (must be freed after MPI_Wait)
}
