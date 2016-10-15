#pragma once
#include <mpich/mpi.h>
#include "utils.h"

class MPIHandler{
public:
	//! Takes &argc, &argv
	MPIHandler(bool enable, int* argc=NULL, char*** argv=NULL);
  MPIHandler(const MPIHandler&);                    // No copy construct/assign
  MPIHandler& operator=(const MPIHandler&);
	~MPIHandler();

  void makeGrid(const int N);
  void bcast(int* buffer, const int count);
  void scatterMat(int* g, int* rcvSubmat);           // Return value for transition into async calls
  void gatherMat (int* rcvSubmat, int* g);           // Return value for transition into async calls

  void barrier() const { if(!disabled) MPI_Barrier(MPI_COMM_WORLD); }
  int procN() const { return (disabled)? 1: procN_; }
  int rank() const { return (disabled)? 0: rank_; }
  int s_x() const { return (disabled)? 0: s_x_; }
  int s_y() const { return (disabled)? 0: s_y_; }
  int submatStart() const { return (disabled)? 0: submatStarts_[rank_]; }
  xy gridCoord() const { return (disabled)? xy(9,9): gridCoord_; }

  const char disabled;

private:
  void makeTypes(const int n, const int N);

	int rank_, procN_;
  xy gridCoord_, gridSize_;
  bool mpitypesDefined_, gridReady_;
  smart_arr<int> submatStarts_, ones_;
  int s_x_, s_y_;
  MPI_Datatype MPI_TILE, MPI_SUBMAT;
};

