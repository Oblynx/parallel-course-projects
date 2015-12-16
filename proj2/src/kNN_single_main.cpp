//! Single-process for debugging
#include <iostream>
#include "kNNAlgo.h"
using namespace std;

int main(){
  int N=1<<18, Q=1<<10;
  Point3 q[Q];
  Parameters param(3, 0.01, 0.01, 0.1, 100,100,10);
  CubeArray cubeArray(param);
  for(int i=0; i<N; i++) cubeArray.place({xor128(), xor128(), xor128()});
  for(int i=0; i<Q; i++) q[i]= Point3(xor128(), xor128(), xor128());

  Search search(cubeArray, param);
  for(int i=0; i<Q; i++) search.query(q[i]);
  return 0; 
}
