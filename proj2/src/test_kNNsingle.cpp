//! Test kNN algorithm in a single-process environment (MPI comms turned off)
#ifndef __DEBUG__
  #define __DEBUG__
#endif
#include <iostream>
#include "kNNAlgo.h"
using namespace std;

Point3f pointGridGen(const int i, const int perDim){
  Point3f p {(float)(i%perDim)/perDim, (float)((i/perDim)%perDim)/perDim, (float)(i/(perDim*perDim))/perDim};
  printf("[gen]: (%f,%f,%f)\n",p.x,p.y,p.z);
  return p;
}

int main(){
  MPIhandler mpi(false);    //MPI turned off
  int mesh=1<<3, nDim=1<<2, N= nDim*nDim*nDim; 
  Parameters param(16,0, mesh,mesh,mesh);
  CubeArray cubeArray(param,0,0,0);
  printf("[test_kNN]: Constructed CubeArray\n");
  for(int i=0; i<N; i++) cubeArray.place(pointGridGen(i,nDim));
  printf("[test_kNN]: Points placed in cube:\n");
  //printf("\t->(%d,%d,%d)\n", n1.x,n1.y,n1.z);
  
  vector<Point3f> q(5);
  vector<deque<Element>> qres;
  q[0]= {0.34234, 0.2346, 0.24546};
  q[1]= {0.4124, 0.413, 0.4128};
  q[2]= {0.1, 0.8, 0.5};
  q[3]= {0.000001, 0.000001, 0.000001};
  q[4]= {0.000001, 0.999999, 0.366};
  printf("[test_kNN]: Queries produced\n");
  Search search(cubeArray, param,mpi);
  for(unsigned i=0; i<q.size(); i++) qres.push_back(search.query(q[i]));
  for(unsigned i=0; i<q.size(); i++){
    printf("[test_kNNsingle]: NN for (%f, %f, %f):\n", q[i].x, q[i].y, q[i].z);
    for(auto&& elt : qres[i]) printf("\t-> (%f,%f,%f): %e\n", elt.x,elt.y,elt.z,sqrt(elt.dist(q[i])));
    printf("\n");
  }
  return 0; 
}
