#include <math.h>
#include <iostream>
#include "kNNAlgo.h"
#include "mpi_transfers.h"
using namespace std;

//! Generates random points + corresponding *CubeArray* (aka proc) address
PointAddress pointGenerator(const Parameters& param){
  //PointAddress p{{xor128(),xor128(),xor128()},{0},0};
  PointAddress p{{(float)rand()/RAND_MAX,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX},{0},0};
  float cdxf,cdyf,cdzf; 
  //Divide coord by CubeArray length (=_CubeL*_CubeArr)
  //Integral part: CubeArray coordinate
  //Fractional pt: whether neighbors should be included
  float frx= modf(p.p.x/(param.xCubeL*param.xCubeArr),&cdxf),
        fry= modf(p.p.y/(param.yCubeL*param.yCubeArr),&cdyf),
        frz= modf(p.p.z/(param.zCubeL*param.zCubeArr),&cdzf);
  int cdx= (int)cdxf, cdy= (int)cdyf, cdz= (int)cdzf;
  //Address of containing CubeArray
  p.address[p.addrUsed++]= cdx+ cdy*param.xArrGl+ cdz*param.yArrGl*param.xArrGl;
  //Addresses of neighboring CubeArrays where the point belongs due to overlap
  if(frx > 1-param.xOverlap && cdx+1 < param.xArrGl) p.address[p.addrUsed++]= p.address[0]+1;
  if(fry > 1-param.yOverlap && cdy+1 < param.yArrGl) p.address[p.addrUsed++]= p.address[0]+param.xArrGl;
  if(frz > 1-param.zOverlap && cdz+1 < param.zArrGl) p.address[p.addrUsed++]= p.address[0]+param.xArrGl*param.yArrGl;
  if(frx < param.xOverlap && cdx-1 >= 0) p.address[p.addrUsed++]= p.address[0]-1;
  if(fry < param.yOverlap && cdy-1 >= 0) p.address[p.addrUsed++]= p.address[0]-param.xArrGl;
  if(frz < param.zOverlap && cdz-1 >= 0) p.address[p.addrUsed++]= p.address[0]-param.xArrGl*param.yArrGl;
  return p;
}
//! Generates random points and assigns them only to the CubeArray that contains them
PointAddress queryGenerator(const Parameters& param){
  //PointAddress p {{xor128(),xor128(),xor128()},{0},1};
  PointAddress p{{(float)rand()/RAND_MAX,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX},{0},0};
  p.address[p.addrUsed++]=  (int)(p.p.x/(param.xCubeL*param.xCubeArr))+
                            (int)(p.p.y/(param.yCubeL*param.yCubeArr))*param.xArrGl+
                            (int)(p.p.z/(param.zCubeL*param.zCubeArr))*param.yArrGl*param.xArrGl;
  return p;
}

int main(int argc, char** argv){
  MPIhandler mpi(&argc, &argv);
  const int N=1<<22, Q=1<<12, P= mpi.procN(), rank= mpi.rank();
  PRINTF("#%d: MPI handler constructed, procN=%d\n",mpi.rank(),P);
  mpi.barrier();
  //TODO: {x,y,z}ArrGl as function of P? (or simply input?)
  Parameters param(5,0,1, 10,10,10, 2,1,1);
  //Different random seed for each process
  std::hash<std::string> hasher;
  int seed= hasher(std::to_string(mpi.rank()))%(1<<20);
  seed= (seed<0)? -seed: seed;
  srand(seed);
  //Generate N/P points
  All2allTransfer pointTransfer(pointGenerator,param,mpi,N/P,P);
  COUT << "#"<<mpi.rank()<<": Points comm started\n";
  mpi.barrier();
  //Generate Q/P queries
  All2allTransfer queryTransfer(queryGenerator,param,mpi,Q/P,P);
  COUT << "#"<<mpi.rank()<<": Queries comm started\n";
  
  //Sync points
  int ptsN;
  auto points= pointTransfer.get(ptsN);
/*!!!*/mpi.barrier();
  CubeArray cubeArray(param,rank%param.xArrGl,rank/param.xArrGl,rank/(param.xArrGl*param.yArrGl));   
  for(int i=0; i<ptsN; i++) {
    if(i==445 || i==446) PRINTF("#%d -- %d/%d, (%f,%f,%f)\n",mpi.rank(),i,ptsN,points[i].x,points[i].y,points[i].z);
    cubeArray.place(points[i]);
  }
  COUT<<"#"<<mpi.rank()<<": All points received\n";

  //Sync queries
  int qN;
  auto queries= queryTransfer.get(qN);
  COUT<<"#"<<mpi.rank()<<": All queries received\n";
  mpi.barrier();

  //Start search
  COUT<<"#"<<mpi.rank()<<": Starting search\n";
  Search search(cubeArray, param, mpi);
  for(int i=0; i<qN; i++) search.query(queries[i]);

  //Test
  COUT<<"#"<<mpi.rank()<<": Testing\n";
  Point3f testQ;
  if(!mpi.rank()) testQ= {0.2,0.5,0.5};
  else testQ= {0.8,0.5,0.5};
  auto qres= search.query(testQ);
  printf("NN for (%f, %f, %f):\n", testQ.x, testQ.y, testQ.z);
  for(auto&& elt : qres)
    printf("\t-> (%f,%f,%f): d= %e\n", elt->x,elt->y,elt->z,elt->d(testQ));
  printf("\n");
  return 0; 
}
