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
void MPIhandler::bcast(const int* buffer, const int count){
  error= MPI_Bcast(&buffer, count, MPI_INT, rank(), MPI_COMM_WORLD); errorHandler();
}

int MPIhandler::scatterTiles(const int* buffer, const int* counts, const int* offsets, int* rcvBuf, const int rcvCount){
  // Generate request key
  int key= rand();
  {
    MPI_Request rq;
    auto storeResult= rqStore_.emplace(make_pair(key,rq));
    while(!storeResult.second){       // While insert is unsuccessful, because key already exists
      key= rand();
      storeResult= rqStore_.emplace(make_pair(key,rq));
    }
  }
  MPI_Request& rq= rqStore_[key];

  MPI_Iscatterv(buffer, counts, offsets, MPI_TILE, rcvBuf, rcvCount, MPI_TILE, 0, MPI_COMM_WORLD, &rq);
  return key;
}
bool MPIhandler::testRq(const int hash){
  int complete= 0;
  error= MPI_Test(&rqStore_[hash], &complete, MPI_STATUS_IGNORE); errorHandler();
  return complete;
}
void MPIhandler::waitRq(const int hash){
  error= MPI_Wait(&rqStore_[hash], MPI_STATUS_IGNORE); errorHandler();
}
