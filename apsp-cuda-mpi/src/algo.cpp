#include <ctime>
#include "utils.h"
#include "cuda_utils.h"
#include "mpi_handler.h"
#include "algo.h"
#include "kernel_wrp.h"

using namespace std;

#define n MAX_THRperBLK2D

//! Compute APSP in g; for rank!=0, g is NULL
double floydWarshall_gpu_mpi(int *g, int N, MPIHandler& mpi, CUDAHandler& cuda){
  const int B= N/n;
  mpi.makeGrid(n,N);
  smart_arr<int> submat(mpi.s_x()*mpi.s_y());

  double begin= clock();
  mpi.scatterMat(g, submat.get());
  loopTiles(submat.get(), B, N, mpi,cuda);
  mpi.gatherMat(submat.get(), g);
  double end= clock() - begin;

  return end;
}

// sg: submat of g
void loopTiles(int* sg, const int B, const int N, MPIHandler& mpi, CUDAHandler& cuda){
  // Allocate message buffers
  HPinPtr<int> tilebuf(n*n), rowbuf(n*mpi.s_x()), colbuf(n*mpi.s_y());
  DPtr<int> d_tile(n,n), d_row(mpi.s_x(),n), d_col(mpi.s_y(),n);
  // Allocate and copy submat
  DPtr<int> dsg(mpi.s_x(), mpi.s_y());       // dsg: device submat of g
  dsg.copyH2D(sg, mpi.s_x(), mpi.s_x(), mpi.s_y());

  for(int b=0; b<B; b++){
    int rcFlag= 0;    // If row or column have executed
    execPhase1(dsg, b,N, tilebuf, d_tile, mpi);
    execPhase2Row(dsg, b,N, d_tile, rowbuf, d_row, mpi,rcFlag);
    execPhase2Col(dsg, b, d_tile, colbuf, d_col, mpi, rcFlag);
    execPhase3(dsg, b, d_row, d_col, mpi, rcFlag);
  }
  dsg.copyD2H(sg, mpi.s_x(), mpi.s_x(), mpi.s_y());
}

//! 1. If current primary tile belongs to this process, execute phase1
//  2. Broadcast tilebuf
//  3. Copy tilebuf to gpu
//  After: everybody has updated dsg and d_tilebuf 
void execPhase1 (DPtr<int>& dsg, const int b,const int N, int* tilebuf, DPtr<int>& d_tile, MPIHandler& mpi){
  const int executor= phase1FindExecutor(b,mpi);
  if( executor == mpi.rank() ) {
    const dim3 bs(n,n);
    int tileStart= (N*b+b)*n - mpi.submatStart(); // local coord = global - submatStart
    phase1( 1,bs, dsg, tileStart, dsg.pitch_elt() );
    dsg.copyD2H(tilebuf, n, n,n, tileStart);
    //for(int i=0; i<n; i++) dsg.copyD2H(tilebuf+n*i, n, N*i);
    cudaStreamSynchronize(cudaStreamPerThread);
  }
  mpi.bcast(tilebuf,n*n, executor);
  d_tile.copyH2D(tilebuf, n, n,n);
}

void execPhase2Row (DPtr<int>& dsg, const int b,const int N, DPtr<int>& d_tile, int* rowbuf,
    DPtr<int>& d_row, MPIHandler& mpi, int& rcFlag){
  const int gridCol= mpi.gridCoord().x;
  const int executor= phase2RowFindExecutor(b, gridCol, mpi);
  if( executor == mpi.rank() ){
    rcFlag|= 1;
    const dim3 bs(n,n), gs(mpi.s_x()/n,1);
    int rowStart= N*n*b - mpi.submatStart();  // local coord = global - submatStart
    phase2Row( gs,bs, dsg, d_tile, rowStart, dsg.pitch_elt() );
    dsg.copyD2H(rowbuf, mpi.s_x(), mpi.s_x(), n, rowStart);
    cudaStreamSynchronize(cudaStreamPerThread);
  }
  mpi.bcastCol(rowbuf, n*mpi.s_x(), executor);
  d_row.copyH2D(rowbuf, mpi.s_x(), mpi.s_x(), n);
}
void execPhase2Col (DPtr<int>& dsg, const int b, DPtr<int>& d_tile, int* colbuf,
    DPtr<int>& d_col, MPIHandler& mpi, int& rcFlag){
  const int gridRow= mpi.gridCoord().y;
  const int executor= phase2ColFindExecutor(b, gridRow, mpi);
  if( executor == mpi.rank() ){
    rcFlag|= 2;
    const dim3 bs(n,n), gs(mpi.s_y()/n,1);
    int colStart= n*b - mpi.submatStart();  // local coord = global - submatStart
    phase2Col( gs,bs, dsg, d_tile, colStart, dsg.pitch_elt() );
    dsg.copyD2H(colbuf, mpi.s_y(), n, mpi.s_y(), colStart);   // WARNING: In d_col, tiles are spread in 1 row!
    //transposeColbuf(colbuf, mpi.s_y());
    cudaStreamSynchronize(cudaStreamPerThread);
  }
  mpi.bcastRow(colbuf, n*mpi.s_y(), executor);
  d_col.copyH2D(colbuf, mpi.s_y(), mpi.s_y(), n);
}

void execPhase3 (DPtr<int>& dsg, const int b, DPtr<int>& d_row, DPtr<int>& d_col, MPIHandler& mpi, int rcFlag){
  // Grid size minus 1 if this submat has the current primary row or col
  const dim3 bs(n,n), gs(mpi.s_x()/n - ((0x0002&rcFlag)>>1), mpi.s_y()/n - (0x0001&rcFlag));
  xy gCd= mpi.gridCoord();
  phase3( gs,bs, dsg, d_row,d_col, b, gCd.x*mpi.s_x(), gCd.y*mpi.s_y(), dsg.pitch_elt(), d_row.pitch_elt(), rcFlag );
}


//! Return rank of process that will calculate the relevant data
int phase1FindExecutor(const int b, MPIHandler& mpi){
  xy gridCd= mpi.tile2grid(xy(b,b));
  return mpi.gridSize().x*gridCd.y + gridCd.x;
  //return mpi.gridCoord() == mpi.tile2grid(xy(b,b));
}
int phase2RowFindExecutor(const int b, const int col, MPIHandler& mpi){
  const int gridY= mpi.tile2grid(xy(b,b)).y;
  return mpi.gridSize().x*gridY + col;
  //return mpi.gridCoord().y == mpi.tile2grid(xy(b,b)).y;
}
int phase2ColFindExecutor(const int b, const int row, MPIHandler& mpi){
  const int gridX= mpi.tile2grid(xy(b,b)).x;
  return mpi.gridSize().x*row + gridX;
  //return mpi.gridCoord().x == mpi.tile2grid(xy(b,b)).x;
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
