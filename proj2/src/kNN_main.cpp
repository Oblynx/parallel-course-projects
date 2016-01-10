#include <cstdio>
#include <math.h>
#include <iostream>
#include <cfloat>
#include <chrono>
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

int PGlob;

//! Generates random points + corresponding *CubeArray* (aka proc) address
PointAddress pointGenerator(const Parameters& param){
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
             p.address[p.addrUsed++]= p.address[0]-1-yoffset-zoffset;
             break;    //-x-y-z
  }
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

unsigned log2floor(unsigned a){
  unsigned b=0;
  if(a!=0) while(a!=1) a>>=1, b++;
  return b;
}

int main(int argc, char** argv){
  MPIhandler mpi(true, &argc, &argv);
  unsigned k=3, N=1<<24, Q=1<<18, nmk=1<<12, P= mpi.procN(), rank= mpi.rank();
PGlob= P;
  if(argc>=3){
    k=   atoi(argv[1]);
    N=   1<<atoi(argv[2]), Q=N;
    nmk= 1<<atoi(argv[3]);
  }
  const unsigned lognmk= log2floor(nmk), cx= lognmk/3, cy= (lognmk%3!=2)? lognmk/3: lognmk/3+1,
        cz= (lognmk%3==0)? lognmk/3: lognmk/3+1;
  const unsigned logP= log2floor(P), px= logP/3, py= (logP%3!=2)? logP/3: logP/3+1,
        pz= (logP%3==0)? logP/3: logP/3+1;
#ifdef BATCH
  FILE *logfile, *resultfile;
  if(!rank){
    int serial= (argc==4)? atoi(argv[4]): 0;
    string logname= "../logs/log"+to_string(serial);
    logfile= fopen(logname.c_str(), "a");
  #ifdef BATCH_RESULTS
    string resultname= "../logs/results"+to_string(serial);
    resultfile= fopen(resultname.c_str(), "a");
  #endif
  }
#endif
  if(!rank) PRINTF("*LOGARITHMIC* Cubes: (%d,%d,%d)\tProcs: (%d,%d,%d)\tEst points/cube=%d\n\n",cx,cy,cz, px,py,pz, N/nmk);
  Parameters param(k,2, 1<<cx,1<<cy,1<<cz, 1<<px,1<<py,1<<pz, N/nmk);
  //Different random seed for each process
  std::hash<std::string> hasher;
  int seed= 1+hasher(std::to_string(mpi.rank()))%(1<<20);
  seed= (seed<0)? -seed: seed;
  srand(seed);

  //Generate N/P points & Q/P queries
  All2allTransfer pointTransfer(pointGenerator,param,mpi,N/P,P);
  All2allTransfer queryTransfer(queryGenerator,param,mpi,Q/P,P);
  auto startTime= chrono::system_clock::now();
  pointTransfer.transfer();   // Request points transfer
  PRINTF("#%d: Points comm started\n",mpi.rank());
  queryTransfer.transfer();   // Request queries transfer
  PRINTF("#%d: Queries comm started\n",mpi.rank());
  
  //Sync -- Actually get the points from nonblocking communications
  int ptsN;
  auto points= pointTransfer.get(ptsN);
  auto gotPointsTime= chrono::system_clock::now();
  PRINTF("#%d: All points received\n",mpi.rank());
  CubeArray cubeArray(param,rank%param.xArrGl, (rank/param.xArrGl)%param.yArrGl, rank/(param.xArrGl*param.yArrGl));

  for(int i=0; i<ptsN; i++) cubeArray.place(points[i]);
  PRINTF("#%d: Points placed in CubeArray\n",mpi.rank());

  //Sync queries
  int qN;
  auto queries= queryTransfer.get(qN);
  auto commsCompleteTime= chrono::system_clock::now();
  PRINTF("#%d: All queries received\n",mpi.rank());
  //mpi.barrier();

  //unique_ptr<std::chrono::duration<double>[]> queryTimes(new std::chrono::duration<double>[qN]);
  //Start search
  PRINTF("#%d: Starting search\n",mpi.rank());
  Search search(cubeArray, param, mpi);
  for(int i=0; i<qN; i++){
    //auto qstart= chrono::system_clock::now();
    auto resultQuery= search.query(queries[i]);
    //queryTimes[i]= chrono::system_clock::now()-qstart;
#ifdef BATCH_RESULTS
    string knn;
    knn+= to_string(mpi.rank())+";"+to_string(queries[i].x)+","+to_string(queries[i].y)+","+to_string(queries[i].z)+";";
    for(auto&& elt : resultQuery)
      knn+= to_string(elt.x)+","+to_string(elt.y)+","+to_string(elt.z)+";"+to_string(sqrt(elt.dist(queries[i])))+";;";
    fprintf(resultfile,"%s\n",knn.c_str());
#else
    /*string knn;
    knn+= "NN#"+to_string(mpi.rank())+" for ("+to_string(queries[i].x)+","+to_string(queries[i].y)+","+to_string(queries[i].z)+"):\n";
    for(auto&& elt : resultQuery)
      knn+= "\t-> ("+to_string(elt.x)+","+to_string(elt.y)+","+to_string(elt.z)+"): d= "+to_string(sqrt(elt.dist(queries[i])))+"\n";
    printf("%s\n", knn.c_str());*/
#endif
  }
  PRINTF("#%d: Search complete!\n",rank);
  auto searchComplete= chrono::system_clock::now();

  chrono::duration<double> ptransferT= gotPointsTime-startTime, ccommsT= commsCompleteTime-startTime,
      searchT= searchComplete-commsCompleteTime;

#ifdef BATCH
  if(!rank){
    fprintf(logfile,"%d;%d,%d,%d;%f,%f,%f\n",rank, atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),
                    ptransferT.count(), ccommsT.count(), searchT.count());
  #if defined(BATCH_RESULTS)
    fprintf(resultfile,"-1;-1,-1,-1;-1,-1,-1;-1;-1;-1,-1,-1;-1;-1;-1,-1,-1;-1;-1;\n\n");
    fclose(resultfile);
  #endif
    fclose(logfile);
  }
#else
  if(!rank) printf("#%d: Point transfer time (s): %f\nComplete comms time (s): %f\nSearch time (s): %f\n",
                   rank, ptransferT.count(), ccommsT.count(), searchT.count());
#endif
  //for(int i=0; i<qN; i++){/*TODO: Print individual searches time for histogram*/}
  PRINTF("#%d: Exit\n",rank);
  return 0; 
}
