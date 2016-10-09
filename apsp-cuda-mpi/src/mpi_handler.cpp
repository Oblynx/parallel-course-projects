#include <limits.h>
#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <cmath>
#include "mpi_handler.h"
using namespace std;

MPIhandler::MPIhandler(bool enable, int* argc, char*** argv): disabled(!enable) {
  if(enable){
    error= MPI_Init(argc, argv);  errorHandler();
    error= MPI_Comm_size(MPI_COMM_WORLD, &procN_);  errorHandler();
    error= MPI_Comm_rank(MPI_COMM_WORLD, &rank_);   errorHandler();
    // https://www.rc.colorado.edu/sites/default/files/Datatypes.pdf
    ones_.reset(new int[procN_]);
    for(int i=0; i<procN_; i++) ones_[i]= 1;
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
void MPIhandler::bcast(int* buffer, const int count){
  error= MPI_Bcast(&buffer, count, MPI_INT, 0, MPI_COMM_WORLD); errorHandler();
}

void MPIhandler::makeTypes(const int n, const int N){
  if(mpitypesDefined_) {
    error= MPI_Type_free(&MPI_TILE); errorHandler();
    error= MPI_Type_free(&MPI_SUBMAT); errorHandler();
  }
  error= MPI_Type_vector(n,n,N, MPI_INT, &MPI_TILE); errorHandler();
  error= MPI_Type_create_resized(MPI_TILE, 0,n*sizeof(int), &MPI_TILE); errorHandler();
  error= MPI_Type_commit(&MPI_TILE); errorHandler();

  error= MPI_Type_vector(submatRowN_,submatRowL_,N, MPI_INT, &MPI_SUBMAT); errorHandler();
  error= MPI_Type_create_resized(MPI_SUBMAT, 0,submatRowL_*sizeof(int), &MPI_SUBMAT); errorHandler();
  error= MPI_Type_commit(&MPI_SUBMAT); errorHandler();
  mpitypesDefined_= true;
}

int MPIhandler::scatterMat(const int* g, int* rcvSubmat){
  if(!matSplit_) throw new std::logic_error("Fist split matrix before calling scatterMat!\n");
/*
  printf("[scatter]: matrix:\n");
  for(int i=0; i<32; i++){
    for(int j=0; j<32; j++)
      printf("%3d ", g[i*32+j]);
    printf("\n");
  }
  printf("\n[scatter]: counts:\n");
  for(int i=0; i<procN_; i++) printf("%3d ", ones_[i]);
  printf("\n[scatter]: Starts:\n");
  for(int i=0; i<procN_; i++) printf("g[%3d]=%3d ", submatStarts_[i], g[submatStarts_[i]]);
  printf("\n");
*/
  MPI_Scatterv(g, ones_.get(), submatStarts_.get(), MPI_SUBMAT, rcvSubmat,1,MPI_SUBMAT, 0,MPI_COMM_WORLD);
  return 0;
}
int MPIhandler::gatherMat(const int* rcvSubmat, int* g){
  if(!matSplit_) throw new std::logic_error("Fist split matrix before calling scatterMat!\n");
  MPI_Gatherv(rcvSubmat,1,MPI_SUBMAT, g, ones_.get(), submatStarts_.get(), MPI_SUBMAT, 0,MPI_COMM_WORLD);
  return 0;
}

void MPIhandler::splitMat(const int N){
  const int p= static_cast<int>(floor(log2(procN_)))/2, y= static_cast<int>(floor(log2(procN_)))%2;
  submatRowL_= N/(p+y+1), submatRowN_= N/(p+1);
  const int usedProcs= 1<<static_cast<int>(floor(log2(procN_)));
  submatStarts_.reset(new int[procN_]);
  for(int p=usedProcs; p<procN_; p++) submatStarts_[p]= 0;
  for(int p=0; p<usedProcs; p++) submatStarts_[p]= (p*submatRowL_)%N + (p*submatRowL_)/N * submatRowN_;
  matSplit_= true;
}


bool MPIhandler::testRq(const int hash){
  int complete= 0;
  error= MPI_Test(&rqStore_[hash], &complete, MPI_STATUS_IGNORE); errorHandler();
  return complete;
}
void MPIhandler::waitRq(const int hash){
  error= MPI_Wait(&rqStore_[hash], MPI_STATUS_IGNORE); errorHandler();
}

// Async comm
/*
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
*/
