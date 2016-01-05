//! Test kNN algorithm in a single-process environment (MPI comms turned off)
#ifndef __DEBUG__
  #define __DEBUG__
#endif
#include <iostream>
#include "kNNAlgo.h"
using namespace std;

int main(){
  MPIhandler mpi(0);    //MPI turned off
  int N=1<<20, Q=1<<16;
  Point3f q[Q];
  Parameters param(5,0, 10,10,10);
  CubeArray cubeArray(param,0,0,0);
  for(int i=0; i<N; i++) cubeArray.place({xor128(), xor128(), xor128()});
  auto n1= cubeArray.place({0.412, 0.412, 0.413});
  auto n2= cubeArray.place({0.413, 0.412, 0.412});
  auto n3= cubeArray.place({0.411, 0.416, 0.413});
  printf("[test_kNN]: Points placed in cube:\n");
  printf("\t->(%d,%d,%d)\n", n1.x,n1.y,n1.z);
  for(int i=0; i<Q-1; i++) q[i]= {xor128(), xor128(), xor128()};
  q[Q-1]= {0.4124, 0.413, 0.4128};
  printf("[test_kNN]: Queries produced\n");
  Search search(cubeArray, param,mpi);
  for(int i=0; i<Q-1; i++) search.query(q[i]);
  auto qres= search.query(q[Q-1]);
  printf("[test_kNN]: 3NN for Point3(%f, %f, %f):\n", q[Q-1].x, q[Q-1].y, q[Q-1].z);
  for(auto&& elt : qres) printf("\t-> (%f,%f,%f)\n", elt.x,elt.y,elt.z);
  printf("\n");
  return 0; 
}
