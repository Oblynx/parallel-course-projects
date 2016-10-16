#pragma once
#include <mpich/mpi.h>
#include "utils.h"

class MPIHandler{
public:
	//! Takes &argc, &argv
	MPIHandler(int* argc, char*** argv);
  MPIHandler(const MPIHandler&);                    // No copy construct/assign
  MPIHandler& operator=(const MPIHandler&);
	~MPIHandler();

  void makeGrid(const int n, const int N);
  void bcast(int* buffer, const int count, const int broadcaster);
  void bcastRow(int* buffer, const int count, const int broadcaster);
  void bcastCol(int* buffer, const int count, const int broadcaster);
  void scatterMat(int* g, int* rcvSubmat);           // Return value for transition into async calls
  void gatherMat (int* rcvSubmat, int* g);           // Return value for transition into async calls

  void barrier() const { MPI_Barrier(MPI_COMM_WORLD); }
  int procN() const { return procN_; }
  int rank() const { return rank_; }
  int s_x() const { return s_x_; }
  int s_y() const { return s_y_; }
  int submatStart() const { return submatStarts_[rank_]; }
  xy subStartXY() const { return subStartXY_[rank_]; }
  xy gridCoord() const { return gridCoord_; }
  xy gridSize() const { return gridSize_; }
  xy tile2grid(xy tileCd) const { return  tileCd/xy(s_x_,s_y_); }

private:
  void makeTypes(const int n, const int N);

	int rank_, rankRow_, rankCol_, procN_;
  xy gridCoord_, gridSize_;
  bool mpitypesDefined_, gridReady_;
  smart_arr<int> submatStarts_, ones_;
  smart_arr<xy> subStartXY_;
  int s_x_, s_y_;
  MPI_Datatype MPI_TILE, MPI_SUBMAT;
  MPI_Comm MPI_COMM_COL, MPI_COMM_ROW;
};

