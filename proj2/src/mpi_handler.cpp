#include "mpi_handler.h"

MPIhandler::MPIhandler(int* argc, char*** argv){
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
std::future<void> MPIhandler::IsendCoordinates(Point3 cd, int n, int dest){
  //TODO
}
