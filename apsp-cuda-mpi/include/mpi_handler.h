#pragma once
#include <mpich/mpi.h>
#include "utils.h"

class MPIhandler{
public:
	//! Takes &argc, &argv
	MPIhandler(bool enable, int* argc=NULL, char*** argv=NULL);
  //MPIhandler(const MPIhandler&) =delete;            // No copy construct/assign
  //MPIhandler& operator=(const MPIhandler&) =delete;
	~MPIhandler();

  void errorHandler();
  void barrier() {
    if(!disabled){
      error= MPI_Barrier(MPI_COMM_WORLD); errorHandler();
    }
  }
  int procN() { return (disabled)? 1: procN_; }
  int rank() { return (disabled)? 0: rank_; }
  int submatRowL() { return (disabled)? 0: submatRowL_; }
  int submatRowN() { return (disabled)? 0: submatRowN_; }
  int submatStart() { return (disabled)? 0: submatStarts_[rank_]; }
  void makeTypes(const int n, const int N);
  void splitMat(const int N);
  void bcast(int* buffer, const int count);
  int scatterMat(int* g, int* rcvSubmat);           // Return value for transition into async calls
  int gatherMat (int* rcvSubmat, int* g);           // Return value for transition into async calls

  const char disabled;
  MPI_Datatype MPI_TILE, MPI_SUBMAT;
private:
	int error, rank_, procN_, mpitypesDefined_, matSplit_;
  int *submatStarts_, *ones_;
  int submatRowL_, submatRowN_;
};


