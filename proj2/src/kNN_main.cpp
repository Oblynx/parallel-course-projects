#include <math.h>
#include <iostream>
#include <cfloat>
#include "kNNAlgo.h"
#include "mpi_transfers.h"
using namespace std;

PointAddress createRand(){
  PointAddress p{{(float)rand()/RAND_MAX,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX},{0},0};
  if(1.0-p.p.x <= FLT_EPSILON) p.p.x-= FLT_EPSILON;
  if(1.0-p.p.y <= FLT_EPSILON) p.p.y-= FLT_EPSILON;
  if(1.0-p.p.z <= FLT_EPSILON) p.p.z-= FLT_EPSILON;
  if(p.p.x <= FLT_EPSILON) p.p.x+= FLT_EPSILON;
  if(p.p.y <= FLT_EPSILON) p.p.y+= FLT_EPSILON;
  if(p.p.z <= FLT_EPSILON) p.p.z+= FLT_EPSILON;
  return p;
}

const int sN=100;
float sx=0,sy=0,sz=0;
MPIhandler* mpiGlob;

//! Generates random points + corresponding *CubeArray* (aka proc) address
PointAddress pointGenerator(const Parameters& param){
  /*sx+= 1.0/sN;
  if(sx-1<= FLT_EPSILON) sx=0, sy+= 1.0/sN;
  if(sy-1<= FLT_EPSILON) sy=0, sz+= 1.0/sN;
  PointAddress p{{sx,sy,sz},{0},0};
  if(1.0-p.p.x <= FLT_EPSILON) p.p.x-= FLT_EPSILON;
  if(1.0-p.p.y <= FLT_EPSILON) p.p.y-= FLT_EPSILON;
  if(1.0-p.p.z <= FLT_EPSILON) p.p.z-= FLT_EPSILON;
  if(p.p.x <= FLT_EPSILON) p.p.x+= FLT_EPSILON;
  if(p.p.y <= FLT_EPSILON) p.p.y+= FLT_EPSILON;
  if(p.p.z <= FLT_EPSILON) p.p.z+= FLT_EPSILON;*/
  auto p= createRand();
  float cdxf,cdyf,cdzf; 
  //Divide coord by CubeArray length (=_CubeL*_CubeArr)
  //Integral part: CubeArray coordinate
  //Fractional pt: whether neighbors should be included
  float frx= modf(p.p.x/(param.xCubeL*param.xCubeArr),&cdxf),
        fry= modf(p.p.y/(param.yCubeL*param.yCubeArr),&cdyf),
        frz= modf(p.p.z/(param.zCubeL*param.zCubeArr),&cdzf);
  int cdx= (int)cdxf, cdy= (int)cdyf, cdz= (int)cdzf;
  //neighbors
  int nbor=0, yoffset= param.xArrGl, zoffset= param.xArrGl*param.yArrGl;
  //Address of containing CubeArray
  p.address[p.addrUsed++]= cdx+ cdy*param.xArrGl+ cdz*param.yArrGl*param.xArrGl;
  //Addresses of neighboring CubeArrays where the point belongs due to overlap
  //Add sides
  if(frx > 1-param.xOverlap && cdx+1 < param.xArrGl) nbor|=1, p.address[p.addrUsed++]= p.address[0]+1;
  if(fry > 1-param.yOverlap && cdy+1 < param.yArrGl) nbor|=2, p.address[p.addrUsed++]= p.address[0]+yoffset;
  if(frz > 1-param.zOverlap && cdz+1 < param.zArrGl) nbor|=4, p.address[p.addrUsed++]= p.address[0]+zoffset;
  if(frx < param.xOverlap && cdx-1 >= 0) nbor|=8, p.address[p.addrUsed++]= p.address[0]-1;
  if(fry < param.yOverlap && cdy-1 >= 0) nbor|=16, p.address[p.addrUsed++]= p.address[0]-param.xArrGl;
  if(frz < param.zOverlap && cdz-1 >= 0) nbor|=32, p.address[p.addrUsed++]= p.address[0]-param.xArrGl*param.yArrGl;
  //Add corners (less tedious?)
  switch (nbor){
			//1 zero +1
		case  3: p.address[p.addrUsed++]= p.address[0]+1+yoffset;	break;    //+x+y
		case  5: p.address[p.addrUsed++]= p.address[0]+1+zoffset;	break;    //+x+z
		case 17: p.address[p.addrUsed++]= p.address[0]+1-yoffset;	break;    //+x-y
		case 33: p.address[p.addrUsed++]= p.address[0]+1-zoffset;	break;    //+x-z

		case 10: p.address[p.addrUsed++]= p.address[0]-1+yoffset;	break;    //-x+y
		case 12: p.address[p.addrUsed++]= p.address[0]-1+zoffset;	break;    //-x+z
		case 24: p.address[p.addrUsed++]= p.address[0]-1-yoffset;	break;    //-x-y
		case 40: p.address[p.addrUsed++]= p.address[0]-1-zoffset;	break;    //-x-z

		case  6: p.address[p.addrUsed++]= p.address[0]+yoffset+zoffset;	break;    //+y+z
		case 34: p.address[p.addrUsed++]= p.address[0]+yoffset-zoffset;	break;    //+y-z
		case 20: p.address[p.addrUsed++]= p.address[0]-yoffset+zoffset;	break;    //-y+z
		case 48: p.address[p.addrUsed++]= p.address[0]-yoffset-zoffset;	break;    //-y-z

			//0 zero +4
		case  7:
             p.address[p.addrUsed++]= p.address[0]+1+yoffset;
             p.address[p.addrUsed++]= p.address[0]+yoffset+zoffset;
             p.address[p.addrUsed++]= p.address[0]+zoffset+1;
             p.address[p.addrUsed++]= p.address[0]+1+yoffset+zoffset;
             break;    //+x+y+z
		case 35:	
             p.address[p.addrUsed++]= p.address[0]+1+yoffset;
             p.address[p.addrUsed++]= p.address[0]+yoffset-zoffset;
             p.address[p.addrUsed++]= p.address[0]-zoffset+1;
             p.address[p.addrUsed++]= p.address[0]+1+yoffset-zoffset;
             break;    //+x+y-z
		case 21:	
             p.address[p.addrUsed++]= p.address[0]+1-yoffset;
             p.address[p.addrUsed++]= p.address[0]-yoffset+zoffset;
             p.address[p.addrUsed++]= p.address[0]+zoffset+1;
             p.address[p.addrUsed++]= p.address[0]+1-yoffset+zoffset;
             break;    //+x-y+z
		case 49:	
             p.address[p.addrUsed++]= p.address[0]+1-yoffset;
             p.address[p.addrUsed++]= p.address[0]-yoffset-zoffset;
             p.address[p.addrUsed++]= p.address[0]-zoffset+1;
             p.address[p.addrUsed++]= p.address[0]+1-yoffset-zoffset;
             break;    //+x-y-z
		case 14:	
             p.address[p.addrUsed++]= p.address[0]-1+yoffset;
             p.address[p.addrUsed++]= p.address[0]+yoffset+zoffset;
             p.address[p.addrUsed++]= p.address[0]+zoffset-1;
             p.address[p.addrUsed++]= p.address[0]-1+yoffset+zoffset;
             break;    //-x+y+z
		case 42:	
             p.address[p.addrUsed++]= p.address[0]-1+yoffset;
             p.address[p.addrUsed++]= p.address[0]+yoffset-zoffset;
             p.address[p.addrUsed++]= p.address[0]-zoffset-1;
             p.address[p.addrUsed++]= p.address[0]-1+yoffset-zoffset;
             break;    //-x+y-z
		case 28:	
             p.address[p.addrUsed++]= p.address[0]-1-yoffset;
             p.address[p.addrUsed++]= p.address[0]-yoffset+zoffset;
             p.address[p.addrUsed++]= p.address[0]+zoffset-1;
             p.address[p.addrUsed++]= p.address[0]-1-yoffset+zoffset;
             break;    //-x-y+z
		case 56:	
             p.address[p.addrUsed++]= p.address[0]-1-yoffset;
             p.address[p.addrUsed++]= p.address[0]-yoffset-zoffset;
             p.address[p.addrUsed++]= p.address[0]-zoffset-1;
             p.address[p.addrUsed++]= p.address[0]-1-yoffset+zoffset;
             break;    //-x-y-z
  }

  /*Point3f error {0.105113,0.462703,0.422777};
  if(fabs(p.p.x-error.x) < 0.000001 && fabs(p.p.y-error.y) < 0.000001 && fabs(p.p.z-error.z) < 0.000001){
    string addr;
    for(int i=0; i<p.addrUsed; i++) addr+= to_string(p.address[i])+";";
    PRINTF("zOverlap=%f\t--> cdz=%d\tfrz=%f\n\t    addresses= %s\n", param.zOverlap,cdz,frz,addr.c_str());
  }*/
  return p;
}
//! Generates random points and assigns them only to the CubeArray that contains them
PointAddress queryGenerator(const Parameters& param){
  auto p= createRand();
  p.address[p.addrUsed++]=  (int)(p.p.x/(param.xCubeL*param.xCubeArr))+
                            (int)(p.p.y/(param.yCubeL*param.yCubeArr))*param.xArrGl+
                            (int)(p.p.z/(param.zCubeL*param.zCubeArr))*param.yArrGl*param.xArrGl;
  return p;
}

int main(int argc, char** argv){
  MPIhandler mpi(&argc, &argv);
   int N=sN*sN*sN-1, Q=1<<10, P= mpi.procN(), rank= mpi.rank();
  N=1<<25;
  mpiGlob= &mpi;
  mpi.barrier();
  //TODO: {x,y,z}ArrGl as function of P? (or simply input?)
  Parameters param(5,2, 20,20,20, 2,2,2);
  //Different random seed for each process
  std::hash<std::string> hasher;
  int seed= hasher(std::to_string(mpi.rank()))%(1<<20);
  seed= (seed<0)? -seed: seed;
  srand(seed);

  
  //Generate N/P points
  All2allTransfer pointTransfer(pointGenerator,param,mpi,N/P,P);
  PRINTF("#%d: Points comm started\n",mpi.rank());
  mpi.barrier();
  //Generate Q/P queries
  All2allTransfer queryTransfer(queryGenerator,param,mpi,Q/P,P);
  PRINTF("#%d: Queries comm started\n",mpi.rank());
  
  //Sync points
  int ptsN;
  auto points= pointTransfer.get(ptsN);
  PRINTF("#%d: All points received\n",mpi.rank());
  CubeArray cubeArray(param,rank%param.xArrGl,rank/param.xArrGl,rank/(param.xArrGl*param.yArrGl));   
  mpi.barrier();
  if(mpi.rank()>3) this_thread::sleep_for(chrono::seconds(2*mpi.rank()-2));
  PRINTF("#%d: Continuing...\n",mpi.rank());
  for(int i=0; i<ptsN; i++){
    if(mpi.rank()==4) PRINTF("#%d -- %d/%d, (%f,%f,%f)\n",mpi.rank(),i,ptsN,points[i].x,points[i].y,points[i].z);
    cubeArray.place(points[i]);
  }
  PRINTF("#%d: Points placed in CubeArray\n",mpi.rank());

  //Sync queries
  int qN;
  auto queries= queryTransfer.get(qN);
  PRINTF("#%d: All queries received\n",mpi.rank());
  //mpi.barrier();

  //Start search
  PRINTF("#%d: Starting search\n",mpi.rank());
  Search search(cubeArray, param, mpi);
  for(int i=0; i<qN; i++) search.query(queries[i]);

  //Test
  PRINTF("#%d: Testing\n",mpi.rank());
  Point3f testQ; bool notest=true;
  switch (mpi.rank()){
    case 0: testQ= {0.2,0.4999,0.49999}; break;
    case 1: testQ= {0.8,0.4,0.49999}; break;
    case 2: testQ= {0.2,0.5,0.49999}; break;
    case 3: testQ= {0.8,0.8,0.49999}; break;
    default: notest=true;
  }
  if(!notest){
    auto qres= search.query(testQ);
    string knn;
    knn+= "NN#"+to_string(mpi.rank())+" for ("+to_string(testQ.x)+","+to_string(testQ.y)+","+to_string(testQ.z)+"):\n";
    for(auto&& elt : qres)
      knn+= "\t-> ("+to_string(elt.x)+","+to_string(elt.y)+","+to_string(elt.z)+"): d= "+to_string(sqrt(elt.distStateful(testQ)))+"\n";
    printf("%s\n", knn.c_str());
  }
  return 0; 
}
