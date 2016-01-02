#pragma once
#include <mpi.h>
#include "utils.h"

class MPIhandler{
public:
	//! Takes &argc, &argv
	MPIhandler(int* argc, char*** argv);
  MPIhandler(char): disabled(true) {}
	~MPIhandler();
  std::future<void> IsendCoordinates(Point3 cd, int n, int dest);
private:
  typedef struct { float x,y,z;} c_Point3f;
  typedef struct { int    x,y,z;} c_Point3;
  MPI_Datatype pT, pfT;
	int error, rank_;
  const char disabled=false;
};

