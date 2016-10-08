#pragma once
#include <mpich/mpi.h>
#include <unordered_map>
#include "utils.h"

class MPIhandler{
public:
	//! Takes &argc, &argv
	MPIhandler(bool enable, int* argc=NULL, char*** argv=NULL);
  MPIhandler(const MPIhandler&) =delete;            // No copy construct/assign
  MPIhandler& operator=(const MPIhandler&) =delete;
	~MPIhandler();

  void errorHandler();
  void barrier() {
    if(!disabled){
      error= MPI_Barrier(MPI_COMM_WORLD); errorHandler();
    }
  }
  int procN() { return (disabled)? 1: procN_; }
  int rank() { return (disabled)? 0: rank_; }
  void makeTileType(const int n, const int N, const int extent){
    if(mpitileDefined_) { error= MPI_Type_free(&MPI_TILE); errorHandler(); }
    error= MPI_Type_vector(n,n,N, MPI_INT, &MPI_TILE); errorHandler();
    error= MPI_Type_create_resized(MPI_TILE, 0,extent*sizeof(int), &MPI_TILE); errorHandler();
    error= MPI_Type_commit(&MPI_TILE); errorHandler();
    mpitileDefined_= true;
  }

  void bcast(const int* buffer, const int count);
  int scatterTiles(const int* buffer, const int* counts, const int* offsets, int* rcvBuf, const int rcvCount);
  bool testRq(const int hash);
  void waitRq(const int hash);

  const char disabled;
  MPI_Datatype MPI_TILE;
private:
	int error, rank_, procN_, mpitileDefined_= false;
  std::unordered_map<int,MPI_Request> rqStore_;
};


