#include <cstdio>
#include <memory>
#include <vector>
#include <cuda_runtime_api.h>
#include "utils.h"
#include "DPtr.h"
#include "mpi_handler.h"
#include "kernel_wrp.h"

using namespace std;

// Scatters the section that starts at sectionStart to all procs
void scatterSection(MPIhandler& mpi, vector<int>& rqStore, int* g, int sectionStart, const int rowLength,
    const int rowN, int* tilesProc, int* counts, int* offsets);

void copyPhase1(int* g, DPtr<int>& d_g1, const int b, const int n, const int N, Dir direction);
void copyPhase2(int* g, DPtr<int>& d_g2, const int b, const int n, const int N, Dir direction);
void copyRowcolMsg(int* g, int* msgRowcol, const int b, const int n, const int N);

// GPU block algo
Duration_fsec run_gpu_mpi_master(MPIhandler& mpi, int* g, const int N, const int* groundTruth, FILE* logfile){
  // Constants
  constexpr const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(MAX_THRperBLK2D, MAX_THRperBLK2D);
  if(N<MAX_THRperBLK2D) bs= dim3(N,N);
  
  // MPI transfer setup
  // TODO: Give N to the other procs!
  mpi.makeTypes(n,N);
  mpi.splitMat(N);
  const int s_x= mpi.submatRowL(), s_y= mpi.submatRowN();

  // Allocate GPU memory for 1st,2nd phase
  DPtr<int> d_g1(n*n);        // 1st phase
  DPtr<int> d_g2(2*n*(N-n));  // 2nd phase
  DPtr<int> d_g3(s_x*s_y);
  unique_ptr<int[]> msgRowcol(new int[2*n*N]);
  unique_ptr<int[]> msgSubmat(new int[s_x*s_y]);

  // Compute APSP
  // For every tile
  printf("Block size: %d\n", n);
  printf("Launching GPU block MPI algo with %d primary blocks\n", B);
  auto start= chrono::system_clock::now();
  for(int b=0; b<B; b++){
    // MPI split phase 3
    //vector<int> mpiAsyncRqTickets(4);
    mpi.scatterMat(g, msgSubmat.get());
    PRINTF("[run_gpu]: b=%d matrix scattered\n",b);

    //##### Compute Phase1&2 #####//
    copyPhase1(g,d_g1,b,n,N,Dir::H2D);          // Copy primary tile to GPU
    phase1(1,bs,d_g1,N);                        // Phase 1 kernel
    
    copyPhase2(g,d_g2,b,n,N,Dir::H2D);          // Copy row&col to GPU
    /* TODO: χρειάζεται? */
    cudaStreamSynchronize(cudaStreamPerThread); // Finish phase1, row,col copies
    PRINTF("[run_gpu]: b=%d phase1 complete\n",b);
    phase2(dim3(B-1,2),bs, d_g1,d_g2,N);        // Phase 2 kernel
    
    copyPhase1(g,d_g1,b,n,N,Dir::D2H);          // Copy primary tile to CPU
    cudaStreamSynchronize(cudaStreamPerThread); // Finish phase2,tile
    PRINTF("[run_gpu]: b=%d phase2 complete\n",b);

    copyPhase2(g,d_g2,b,n,N,Dir::D2H);          // Copy row&col to CPU
    cudaStreamSynchronize(cudaStreamPerThread);
    // Phases 1&2 + CPU/GPU transfers complete
    
    //##### MPI tile, row, col #####//
    copyRowcolMsg(g,msgRowcol.get(), b,n,N);         // Copy row&col to CPU
    // MPI bcast row+col
    mpi.bcast(msgRowcol.get(),n*(N-n));
    PRINTF("[run_gpu]: b=%d row/col broadcasted\n",b);
    //for(auto&& rq: mpiAsyncRqTickets) mpi.waitRq(rq);   // Wait for all phase3 transfers to complete
    mpi.barrier();

    //##### Compute Phase3 #####//
    d_g3.copy(msgSubmat.get(),s_x*s_y, Dir::H2D);
    phase3(dim3(s_x/n, s_y/n),bs, d_g3, d_g2, b-mpi.submatStart()/n, s_x);
    d_g3.copy(msgSubmat.get(),s_x*s_y, Dir::D2H);
    cudaStreamSynchronize(cudaStreamPerThread);
    PRINTF("[run_gpu]: b=%d phase3 complete\n",b);

    mpi.gatherMat(msgSubmat.get(),g);
    PRINTF("[run_gpu]: b=%d matrix gathered\n",b);
    mpi.barrier();
  }
  auto GPUBlock_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("GPU block kernel done: %.3f\n", GPUBlock_time.count());
#ifdef LOG
  fprintf(logfile, "%.5f;", GPUBlock_time.count());
#endif
  auto check= test(g, groundTruth, N, "GPUblock");
  if(!check){
    printf("[GPUblock]: Test FAILED!\n");
    exit(1);
  }
  return GPUBlock_time; 
}

/*
// GPU block algo -- multiple vertices per thread (y only)
Duration_fsec run_GPUblock_multiy(MPIhandler& mpi, const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile ){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_block(N*N);
  constexpr const int n= MAX_THRperBLK2D_MULTI;
  const int B= N/n;
  dim3 bs(n, n/2);

  printf("Launching GPU multi2 block algo with %d primary blocks\n", B);
  auto start= chrono::system_clock::now();
  d_g.copy(g.get(), N*N, Dir::H2D);
  for(int b=0; b<B; b++){
    phase1_multiy<n> <<<1,bs>>>(d_g, b*n, N);
    phase2_multiy<n> <<<dim3(B-1,2),bs>>>(d_g, b*n, b, N);
    phase3_multiy<n> <<<dim3(B-1,B-1),bs>>>(d_g, b*n, b, N);
  }
  d_g.copy(result_block.get(), N*N, Dir::D2H);
  auto GPUBlock_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("GPU multi2 block kernel done: %.3f\n", GPUBlock_time.count());
#ifdef LOG
  fprintf(logfile, "%.5f;", GPUBlock_time.count());
#endif
  auto check= test(result_block, groundTruth, N, "GPUblock_multi2");
  if(!check){
    printf("[GPUblock_multi2]: Test FAILED!\n");
    exit(1);
  }
  return GPUBlock_time; 
}
*/


/*#########  Copy helpers  #########*/
void copyPhase1(int* g, DPtr<int>& d_g1, const int b, const int n, const int N, Dir direction){
  // Copy primary tile to GPU
  for(int i=0; i<n; i++){
    d_g1.copy(g+n*b+N*i,n, direction,n*i);
  }
}
void copyPhase2(int* g, DPtr<int>& d_g2, const int b, const int n, const int N, Dir direction){
  // Copy row to GPU
  for(int i=0; i<n; i++){
    d_g2.copy(g+N*b+N*i, n*b, direction, (N-n)*i);    // Row of g - primary tile
    d_g2.copy(g+N*b+N*i+2*n*b, N-n*(b+1), direction, (N-n)*i+n*b);
  }
  // Copy first part of column, before the primary tile, to GPU
  for(int i=0; i<n*b; i++){
    d_g2.copy(g+n*b+N*i, n, direction, (N-n)*n+N*(i%n)+n*(i/n));  // Column of g row by row - primary tile
    // The column is stored tile-by-tile on the GPU. Each tile of the column that is below the previous on the CPU
    // goes next to the previous one on the GPU.
  }
  // Copy the rest of the column
  for(int i=n*(b+1); i<N; i++){
    d_g2.copy(g+n*b+N*i, n, direction, (N-n)*n+N*((i-n)%n)+n*((i-n)/n));
  }
}

void copyRowcolMsg(int* g, int* msgRowcol, const int b, const int n, const int N){
  // TODO: Pick final version
  /*  ### Version: N-n ###
  // Copy row to message
  for(int i=0; i<n; i++){
    for(int j=0; j < n*b; j++){
      msgRowcol[(N-n)*i +j]= g[N*b+N*i +j];
    }
    for(int j=0; j < N-n*(b+1); j++){
      msgRowcol[(N-n)*i+n*b +j]= g[N*b+N*i+n*(b+1) +j];
    }
  }
  // Copy col to message
  for(int i=0; i<n*b; i++){
    for(int j=0; j<n; j++){
      msgRowcol[(N-n)*n + N*(i%n) + n*(i/n) +j]= g[n*b + N*i +j];
    }
  }
  for(int i=n*(b+1); i<N; i++){
    for(int j=0; j<n; j++){
      msgRowcol[(N-n)*n + N*((i-n)%n)+n*((i-n)/n) +j]= g[n*b + N*i +j];
    }
  }
  */

  // ### Version: N ###
  for(int j=0; j<n*N; j++) msgRowcol[j]= g[n*b*N +j];   // Row
  for(int j=0; j<n*N; j++)
    msgRowcol[n*N +j]= g[n*b + j%n + N*(j/n)];           // Col
}


/*#########  MPI comm helpers  #########*/
// Partition row into MPI processes
// Input:
//   - rowStart: Pointer to 1st elt of row
//   - procN: Total processes
// In/Out:
//   - startProc: The first process for which to calculate. Becomes the last process used +1
//   - remainingTilesProc: How many tiles the proc still requires
// Output:
//   - start,count: [for each proc]: params for MPI_Scatterv
//   - return: Row elements remaining (if row is fully covered, =0)
/*int partitionRow(int rowStart, int rowSize, const int procN, int& startProc, int* remainingTilesProc,
    int* start, int* count){
  int p= startProc;       // p: process index
  for(p= startProc; p < procN && rowSize > 0; p++){
    if(remainingTilesProc[p] < rowSize){
      start[p]= rowStart, count[p]= remainingTilesProc[p];
      remainingTilesProc[p]= 0;
      rowStart+= count[p]; rowSize-= count[p];
    } else {
      start[p]= rowStart, count[p]= rowSize;
      remainingTilesProc[p]-= count[p];
      rowSize= 0;
    }
  }
  startProc= p;
  return rowSize;
}*/

// Partitions the section's tiles into processes. Output: start,count (for MPIscatterv)
// Returns the last covered tile in the section
/*int partitionSection(int sectionStart, const int row1Length, const int rowLength, const int rowsN,
    const int procN, int& lastRowReached, int* remainingTilesProc, int* start, int* count){
  int remainingInRow= 0, row= 0, proc= 0; 
  if(row1Length != rowLength){               // If starting with a partial row
    remainingInRow= partitionRow(sectionStart+rowLength*row, row1Length, procN, proc, remainingTilesProc, start, count);
    row++;
  }
  for(; row<rowsN; row++){
    remainingInRow= partitionRow(sectionStart+rowLength*row, rowLength, procN, proc, remainingTilesProc, start, count);
    if(proc == procN){
      row++;
      break;
    }
  }
  lastRowReached+= row-1;
  return row*rowLength - remainingInRow;
}*/

/*void scatterSection(MPIhandler& mpi, vector<int>& rqStore, int* g, int sectionStart, const int rowLength,
    const int rowN, int* tilesProc, int* counts, int* offsets){
  const int sectionSize= rowLength*rowN;
  int partEnd= 0, lastRowReached= 0;
  while(partEnd < sectionSize){      // Section not done
    partEnd+= partitionSection(sectionStart+ partEnd, (lastRowReached+1)*rowLength-partEnd, rowLength,
                               rowN-lastRowReached, mpi.procN(), lastRowReached, tilesProc, offsets, counts);
    rqStore.push_back(mpi.scatterTiles(g+sectionStart, counts, offsets, nullptr,0));
  }
}*/


