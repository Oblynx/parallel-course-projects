#include <limits.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include "mpi_handler.h"
using namespace std;

MPIHandler::MPIHandler(int* argc, char*** argv): mpitypesDefined_(false), gridReady_(false) {
  MPI_Init(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD, &procN_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  // https://www.rc.colorado.edu/sites/default/files/Datatypes.pdf
  submatStarts_.reset(procN_);
  ones_.reset(procN_);
  for(int i=0; i<procN_; i++) ones_[i]= 1;
}
MPIHandler::~MPIHandler() {
  if(mpitypesDefined_){
    MPI_Type_free(&MPI_TILE);
    MPI_Type_free(&MPI_SUBMAT);
    MPI_Comm_free(&MPI_COMM_ROW);
    MPI_Comm_free(&MPI_COMM_COL);
  }
  MPI_Finalize();
}

void MPIHandler::makeGrid(const int n, const int N){
  // Check if procN is a perfect square (equiv, if log divisible by 2)
  const int lgp= static_cast<int>(floor(log2(procN_))), perfectSquare= !(lgp%2);
  const int sqrt= 1<<(lgp/2);   // Sqrt of proc num
  // If perfect square, same division on each dimension. Otherwise, the procN is 2*sqrt^2, then one dim takes
  // 2*sqrt and the other takes sqrt
  gridSize_= (perfectSquare)? xy(sqrt,sqrt): xy(2*sqrt,sqrt);

  s_x_= N/gridSize_.x, s_y_= N/gridSize_.y;
  const int usedProcs= 1<<static_cast<int>(floor(log2(procN_)));

  for(int p=usedProcs; p<procN_; p++) submatStarts_[p]= 0;
  for(int p=0; p<usedProcs; p++) submatStarts_[p]= (p*s_x_)%N + (p*s_x_)/N * s_y_;

  gridCoord_= xy(rank_%gridSize_.x, rank_/gridSize_.x);

  gridReady_= true;
  makeTypes(n, N);
}
void MPIHandler::makeTypes(const int n, const int N){
  if(!gridReady_) throw new std::logic_error("First make grid, then make types!\n");
  if(mpitypesDefined_) {
    MPI_Type_free(&MPI_TILE); 
    MPI_Type_free(&MPI_SUBMAT); 
  }
  MPI_Comm_split(MPI_COMM_WORLD, gridCoord_.x, rank_, &MPI_COMM_COL);
  MPI_Comm_split(MPI_COMM_WORLD, gridCoord_.y, rank_, &MPI_COMM_ROW);
  MPI_Comm_rank(MPI_COMM_COL, &rankCol_);
  MPI_Comm_rank(MPI_COMM_ROW, &rankRow_);

  MPI_Type_vector(n,n,N, MPI_INT, &MPI_TILE); 
  MPI_Type_create_resized(MPI_TILE, 0,n*sizeof(int), &MPI_TILE); 
  MPI_Type_commit(&MPI_TILE); 

  MPI_Type_vector(s_y_,s_x_,N, MPI_INT, &MPI_SUBMAT); 
  MPI_Type_create_resized(MPI_SUBMAT, 0, s_x_, &MPI_SUBMAT); 
  MPI_Type_commit(&MPI_SUBMAT); 
  mpitypesDefined_= true;
  
  int tsize, ssize;
  MPI_Type_size(MPI_TILE,&tsize); MPI_Type_size(MPI_SUBMAT,&ssize);
  printf("[mpi]: n=%d\tN=%d\tsx=%d\tsy=%d\n", n,N, s_x_, s_y_);
  printf("[mpi]: tsize=%d, n*N=%d\tssize=%d, sy*N=%d\n", tsize, n*N, ssize, s_y_*N);
}

void MPIHandler::scatterMat(int* g, int* rcvSubmat){
  if(!gridReady_) throw new std::logic_error("First make grid before calling scatterMat!\n");
  MPI_Scatterv(g, ones_.get(), submatStarts_.get(), MPI_SUBMAT, rcvSubmat,
      s_x_*s_y_,MPI_INT, 0,MPI_COMM_WORLD);
}
void MPIHandler::gatherMat(int* rcvSubmat, int* g){
  if(!gridReady_) throw new std::logic_error("Fist make grid before calling gatherMat!\n");
  MPI_Gatherv(rcvSubmat, s_x_*s_y_,MPI_INT, g, ones_.get(),
      submatStarts_.get(), MPI_SUBMAT, 0,MPI_COMM_WORLD);
}

void MPIHandler::bcast(int* buffer, const int count, const int broadcaster){
  MPI_Bcast(buffer, count, MPI_INT, broadcaster, MPI_COMM_WORLD);
}
void MPIHandler::bcastRow(int* buffer, const int count, const int broadcaster){
  // Convert WORLD rank to ROW rank
  const int rrank= broadcaster - gridSize_.x*gridCoord_.y;
  MPI_Bcast(buffer, count, MPI_INT, rrank, MPI_COMM_ROW);
}
void MPIHandler::bcastCol(int* buffer, const int count, const int broadcaster){
  // Convert WORLD rank to COL rank
  const int crank= (broadcaster - gridCoord_.x)/gridSize_.x;
  MPI_Bcast(buffer, count, MPI_INT, crank, MPI_COMM_COL);
}
