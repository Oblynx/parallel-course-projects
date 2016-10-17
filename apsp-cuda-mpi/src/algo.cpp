#include <ctime>
#include "utils.h"
#include "cuda_utils.h"
#include "mpi_handler.h"
#include "algo.h"
#include "kernel_wrp.h"

using namespace std;

#define n MAX_THRperBLK2D

//! Compute APSP in g; for rank!=0, g is NULL
double floydWarshall_gpu_mpi(const int* truth, int *g, int N, MPIHandler& mpi, CUDAHandler& cuda){
  const int B= N/n;
  mpi.makeGrid(n,N);
  smart_arr<int> submat(mpi.s_x()*mpi.s_y());

  clock_t begin= clock();
  mpi.scatterMat(g, submat.get());
    printf("[rank#%d]: Scattered: sx=%3d sy=%3d\n", mpi.rank(), mpi.s_x(), mpi.s_y());
    printG_force(submat.get(), n,mpi.s_x(),mpi.s_y());
  loopTiles(truth, submat.get(), B, N, mpi,cuda);
  mpi.gatherMat(submat.get(), g);
  double end= (double)(clock() - begin) / CLOCKS_PER_SEC;

  return end;
}

// sg: submat of g
void loopTiles(const int* truth, int* sg, const int B, const int N, MPIHandler& mpi, CUDAHandler& cuda){
  // Allocate message buffers
  HPinPtr<int> tilebuf(n*n), rowbuf(n*mpi.s_x()), colbuf(n*mpi.s_y());
  //DPtr<int> d_tile(n*n), d_row(mpi.s_x()*n), d_col(mpi.s_y()*n);
  DPtr<int> d_tile(n,n), d_row(mpi.s_x(),n), d_col(mpi.s_y()*n);
  // Allocate and copy submat
  DPtr<int> dsg(mpi.s_x(),mpi.s_y());       // dsg: device submat of g
  dsg.copyH2D(sg,mpi.s_x(), mpi.s_x(),mpi.s_y());

  PRINTF("[algo#%d]: Starting main loop\n", mpi.rank());
  printG(sg,n,mpi.s_x(),mpi.s_y());
  for(int b=0; b<B; b++){
    int rcFlag= 0;    // If row or column have executed
    execPhase1(dsg, b,N, tilebuf, d_tile, mpi,cuda);
      dsg.copyD2H(sg,mpi.s_x(), mpi.s_x(),mpi.s_y());
      PRINTF("[algo#%d]: b=%d after ph1:\n", mpi.rank(),b);
      printG(sg,n,mpi.s_x(),mpi.s_y());
    execPhase2Row(dsg, b,N, d_tile, rowbuf, d_row, mpi,cuda, rcFlag);
      dsg.copyD2H(sg,mpi.s_x(), mpi.s_x(),mpi.s_y());
      PRINTF("[algo#%d]: b=%d after ph2r:\n", mpi.rank(),b);
      printG(sg,n,mpi.s_x(),mpi.s_y());
    execPhase2Col(dsg, b, d_tile, colbuf, d_col, mpi,cuda, rcFlag);
      dsg.copyD2H(sg,mpi.s_x(), mpi.s_x(),mpi.s_y());
      PRINTF("[algo#%d]: b=%d after ph2c:\n", mpi.rank(),b);
      printG(sg,n,mpi.s_x(),mpi.s_y());
    execPhase3(dsg, b, d_row, d_col, mpi,cuda, rcFlag);
      //dsg.copyD2H(sg, mpi.s_x()*mpi.s_y());
      dsg.copyD2H(sg,mpi.s_x(), mpi.s_x(),mpi.s_y());
      PRINTF("[algo#%d]: b=%d after ph3:\n", mpi.rank(),b);
      printG(sg,n,mpi.s_x(),mpi.s_y());
  }
  //dsg.copyD2H(sg, mpi.s_x()*mpi.s_y());
  dsg.copyD2H(sg,mpi.s_x(), mpi.s_x(),mpi.s_y());
}

//! 1. If current primary tile belongs to this process, execute phase1
//  2. Broadcast tilebuf
//  3. Copy tilebuf to gpu
//  After: everybody has updated dsg and d_tilebuf 
void execPhase1 (DPtr<int>& dsg, const int b,const int N, int* tilebuf, DPtr<int>& d_tile, MPIHandler& mpi, CUDAHandler& cuda){
  const int executor= phase1FindExecutor(b,mpi);
  if( executor == mpi.rank() ) {
    const dim3 bs(n,n);
    xy tileStart= xy(n*b, n*b) - mpi.subStartXY(); // local coord = global - submatStart
    PRINTF("[algo#%d]: %d Phase1 n=%d\ttstart=(%d,%d)\n",mpi.rank(),b,n, tileStart.x, tileStart.y);
    phase1( 1,bs, dsg, tileStart, dsg.pitch_elt() );
    dsg.copyD2H(tilebuf,n, n,n, tileStart);
    //dsg.copyD2H_multi(tilebuf, n,n, tileStartXY);
    cuda.synchronize();
  }
  mpi.bcast(tilebuf,n*n, executor);
  d_tile.copyH2D(tilebuf,n, n,n);
}

void execPhase2Row (DPtr<int>& dsg, const int b,const int N, DPtr<int>& d_tile, int* rowbuf,
    DPtr<int>& d_row, MPIHandler& mpi,CUDAHandler& cuda, int& rcFlag){
  const int gridCol= mpi.gridCoord().x;
  const int executor= phase2RowFindExecutor(b, gridCol, mpi);
  if( executor == mpi.rank() ){
    PRINTF("[algo#%d]: %d Phase2r\n",mpi.rank(),b);
    rcFlag|= 1;
    const dim3 bs(n,n), gs(mpi.s_x()/n,1);
    xy rowStart= xy(0, n*b - mpi.subStartXY().y);
    phase2Row( gs,bs, dsg, d_tile, rowStart.y, dsg.pitch_elt(), d_tile.pitch_elt() );
    PRINTF("[ph2r#%d]: rstart=(%d,%d) sx=%d\n", mpi.rank(), rowStart.x,rowStart.y, mpi.s_x());
    dsg.copyD2H(rowbuf,mpi.s_x(), mpi.s_x(),n, rowStart);
    //dsg.copyD2H_multi(rowbuf, mpi.s_x(), n, rowStartXY);
    cuda.synchronize();

    PRINTF("[ph2row#%d]: rowbuf: \trowStart=(%d,%d)\n", mpi.rank(), rowStart.x, rowStart.y);
    printG(rowbuf, n,mpi.s_x(),n);
  }
  mpi.bcastCol(rowbuf, n*mpi.s_x(), executor);
  d_row.copyH2D(rowbuf,mpi.s_x(), mpi.s_x(),n);
}

void execPhase2Col (DPtr<int>& dsg, const int b, DPtr<int>& d_tile, int* colbuf,
    DPtr<int>& d_col, MPIHandler& mpi,CUDAHandler& cuda, int& rcFlag){
  const int gridRow= mpi.gridCoord().y;
  const int executor= phase2ColFindExecutor(b, gridRow, mpi);
  if( executor == mpi.rank() ){
    PRINTF("[algo#%d]: %d Phase2c\n",mpi.rank(),b);
    rcFlag|= 2;
    const dim3 bs(n,n), gs(mpi.s_y()/n,1);
    xy colStart= xy(n*b - mpi.subStartXY().x, 0);
    phase2Col( gs,bs, dsg, d_tile, colStart.x, dsg.pitch_elt(), d_tile.pitch_elt() );
    dsg.copyD2H(colbuf,n, n,mpi.s_y(), colStart);
    //dsg.copyD2H_multi(colbuf, n, mpi.s_y(), colStartXY);   // WARNING: In d_col, tiles are spread in 1 row!
    cuda.synchronize();

    PRINTF("[ph2col#%d]: colbuf: \tcolStart=(%d,%d)\n", mpi.rank(), colStart.x,colStart.y);
    printG(colbuf, mpi.s_y(),mpi.s_y(),n);
  }
  mpi.bcastRow(colbuf, n*mpi.s_y(), executor);
  d_col.copyH2D(colbuf, mpi.s_y()*n);
}

void execPhase3 (DPtr<int>& dsg, const int b, DPtr<int>& d_row, DPtr<int>& d_col, MPIHandler& mpi, CUDAHandler& cuda, int rcFlag){
  // Grid size minus 1 if this submat has the current primary row or col
  const dim3 bs(n,n), gs(mpi.s_x()/n - ((0x0002&rcFlag)>>1), mpi.s_y()/n - (0x0001&rcFlag));
  xy gCd= mpi.gridCoord();
  PRINTF("[algo#%d]: %d Phase3\trc=%d gs=(%d,%d) start=(%d,%d)\n",mpi.rank(),b, rcFlag, gs.x,gs.y, 
      gCd.x*mpi.s_x(),gCd.y*mpi.s_y());
  phase3( gs,bs, dsg, d_row,d_col, b, xy(gCd.x*mpi.s_x()/n, gCd.y*mpi.s_y()/n), dsg.pitch_elt(), d_row.pitch_elt(), d_col.pitch_elt(), rcFlag );
  cuda.synchronize();
}


//! Return rank of process that will calculate the relevant data
int phase1FindExecutor(const int b, MPIHandler& mpi){
  xy gridCd= mpi.point2grid(xy(b,b)*n);
  return mpi.gridSize().x*gridCd.y + gridCd.x;
}
int phase2RowFindExecutor(const int b, const int col, MPIHandler& mpi){
  const int gridY= mpi.point2grid(xy(b,b)*n).y;
  return mpi.gridSize().x*gridY + col;
}
int phase2ColFindExecutor(const int b, const int row, MPIHandler& mpi){
  const int gridX= mpi.point2grid(xy(b,b)*n).x;
  return mpi.gridSize().x*row + gridX;
}

//! Change the arrangement of data in the column buffer.
// Upon copy from GPU, each tile is spread linearly on the buffer (each row right next to the next). Make 
void transposeColbuf(int* colbuf, const int s_y){
  smart_arr<int> buf2(n*s_y);
  for(int i=0; i<n*s_y; i++) buf2[i]= colbuf[i];
  for(int b=0; b<s_y/n; b++)
    for(int i=0; i<n; i++)
      for(int j=0; j<n; j++)
        colbuf[n*b+ s_y*i+ j]= buf2[n*n*b+ n*i+ j];
}
