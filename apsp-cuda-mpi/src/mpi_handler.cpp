#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <cmath>
#include "mpi_handler.h"
using namespace std;

MPIhandler::MPIhandler(bool enable, int* argc, char*** argv): disabled(!enable) {
  mpitypesDefined_= false, matSplit_= false;
  if(enable){
    error= MPI_Init(argc, argv);  errorHandler();
    error= MPI_Comm_size(MPI_COMM_WORLD, &procN_);  errorHandler();
    error= MPI_Comm_rank(MPI_COMM_WORLD, &rank_);   errorHandler();
    // https://www.rc.colorado.edu/sites/default/files/Datatypes.pdf
    ones_= new int[procN_];
    for(int i=0; i<procN_; i++) ones_[i]= 1;
  } else printf("MPI disabled\n");
}
MPIhandler::~MPIhandler() {
  if(!disabled){
    MPI_Finalize();
    delete[](ones_);
    delete[](submatStarts_);
  }
}
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

void MPIhandler::splitMat(const int N){
  const int p= static_cast<int>(floor(log2(procN_)))/2, y= static_cast<int>(floor(log2(procN_)))%2;
  submatRowL_= N/(p+y+1), submatRowN_= N/(p+1);
  const int usedProcs= 1<<static_cast<int>(floor(log2(procN_)));
  submatStarts_= new int[procN_];
  for(int p=usedProcs; p<procN_; p++) submatStarts_[p]= 0;
  for(int p=0; p<usedProcs; p++) submatStarts_[p]= (p*submatRowL_)%N + (p*submatRowL_)/N * submatRowN_;
  matSplit_= true;
}

void MPIhandler::makeTypes(const int n, const int N){
  if(!matSplit_) throw new std::logic_error("First split matrix, then make types!\n");
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

int MPIhandler::scatterMat(int* g, int* rcvSubmat){
  if(!matSplit_) throw new std::logic_error("First split matrix before calling scatterMat!\n");
  MPI_Scatterv(g, ones_, submatStarts_, MPI_SUBMAT, rcvSubmat,1,MPI_SUBMAT, 0,MPI_COMM_WORLD);
  return 0;
}
int MPIhandler::gatherMat(int* rcvSubmat, int* g){
  if(!matSplit_) throw new std::logic_error("Fist split matrix before calling scatterMat!\n");
  MPI_Gatherv(rcvSubmat,1,MPI_SUBMAT, g, ones_, submatStarts_, MPI_SUBMAT, 0,MPI_COMM_WORLD);
  return 0;
}

