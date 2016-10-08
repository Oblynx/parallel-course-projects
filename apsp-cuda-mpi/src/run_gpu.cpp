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

// GPU block algo
Duration_fsec run_GPUblock(MPIhandler& mpi, int* g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile ){
  // Constants
  constexpr const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(MAX_THRperBLK2D, MAX_THRperBLK2D);
  if(N<MAX_THRperBLK2D) bs= dim3(N,N);
  
  // MPI transfer setup
  mpi.makeTileType(n,N, n);
  unique_ptr<int[]> tilesPerProc(new int[mpi.procN()] {((B-1)*(B-1))/mpi.procN()}),
                    counts(new int[mpi.procN()]),       // Used for each (MPI scatter)
                    offsets(new int[mpi.procN()+1]);    // Used for each (MPI scatter)
  unique_ptr<int[]> tilesPerProcInit(new int[mpi.procN()]);   // Copy for resetting later
  {
    int remainingTiles= ((B-1)*(B-1))%mpi.procN();
    
    for(int p=0; p<mpi.procN(); p++){
      tilesPerProc[p]+= (remainingTiles-- > 0);
      tilesPerProcInit[p]= tilesPerProc[p];
    }
  }

  // Allocate GPU memory for 1st,2nd phase
  DPtr<int> d_g1(n*n);        // 1st phase
  DPtr<int> d_g2(2*n*(N-n));  // 2nd phase
  unique_ptr<int[]> msgRow(new int[n*(N-n)]), msgCol(new int[n*(N-n)]);

  // Compute APSP
  // For every tile
  for(int b=0; b<B; b++){
    // MPI split phase 3
    vector<int> mpiAsyncRqTickets(4);
    {
      int rowLengths[] {b,B-1-b,b,B-1-b};      // Row length (tiles) in each matrix section
      int rowNs[]      {b,b,B-1-b,B-1-b};      // Row length (tiles) in each matrix section
      int sectionStarts[] {0, n*(b+1), N*(b+1), (N+n)*(b+1)};  // Indices of g where the sections begin

      // Split section A
      scatterSection(mpi,mpiAsyncRqTickets,g, sectionStarts[0], rowLengths[0], rowNs[0],
                     tilesPerProc.get(), counts.get(), offsets.get());
      for(int i=0; i<mpi.procN(); i++) tilesPerProc[i]= tilesPerProcInit[i];
      // Split section B
      scatterSection(mpi,mpiAsyncRqTickets,g, sectionStarts[1], rowLengths[1], rowNs[1],
                     tilesPerProc.get(), counts.get(), offsets.get());
      for(int i=0; i<mpi.procN(); i++) tilesPerProc[i]= tilesPerProcInit[i];
      // Split section C
      scatterSection(mpi,mpiAsyncRqTickets,g, sectionStarts[2], rowLengths[2], rowNs[2],
                     tilesPerProc.get(), counts.get(), offsets.get());
      for(int i=0; i<mpi.procN(); i++) tilesPerProc[i]= tilesPerProcInit[i];
      // Split section D
      scatterSection(mpi,mpiAsyncRqTickets,g, sectionStarts[3], rowLengths[3], rowNs[3],
                     tilesPerProc.get(), counts.get(), offsets.get());
      for(int i=0; i<mpi.procN(); i++) tilesPerProc[i]= tilesPerProcInit[i];
    }

    copyPhase1(g,d_g1,b,n,N,Dir::H2D);        // Copy primary tile to GPU
    phase1(1,bs,d_g1,N);                        // Phase 1 kernel
    
    copyPhase2(g,d_g2,b,n,N,Dir::H2D);        // Copy row&col to GPU
    /* TODO: χρειάζεται? */
    cudaStreamSynchronize(cudaStreamPerThread); // Finish phase1, row,col copies
    phase2(dim3(B-1,2),bs, d_g1,d_g2,N);        // Phase 2 kernel

    // Copy tile to CPU
    for(int i=0; i<n; i++){
      d_g1.copy(g+N*i,n,Dir::D2H,n*i);
    }
    cudaStreamSynchronize(cudaStreamPerThread); // Finish phase2,tile

    // Copy row to CPU
    for(int i=0; i<n; i++){
      d_g2.copy(g+N*b+N*i, n*b,Dir::D2H, (N-n)*i);    // Row of g - primary tile
      d_g2.copy(g+N*b+N*i+n*(b+1), N-n*(b+1),Dir::D2H, (N-n)*i+n*b);
    }
    // Copy column to CPU
    for(int i=0; i<n*b; i++){
      d_g2.copy(g+n*b+N*i, n,Dir::D2H, (N-n)*n+N*(i%n)+n*(i/n));  // Column of g row by row - primary tile
    }
    // Copy the rest of the column
    for(int i=n*(b+1); i<N; i++){
      d_g2.copy(g+n*b+N*i, n,Dir::D2H, (N-n)*n+N*((i-n)%n)+n*((i-n)/n));
    }
    cudaStreamSynchronize(cudaStreamPerThread);
    // Phases 1&2 + CPU/GPU transfers complete
    
    // Copy row to message
    for(int i=0; i<n; i++){
      for(int j=0; j < n*b; j++){
        msgRow[(N-n)*i +j]= g[N*b+N*i +j];
      }
      for(int j=0; j < N-n*(b+1); j++){
        msgRow[(N-n)*i+n*b +j]= g[N*b+N*i+n*(b+1) +j];
      }
    }
    // Copy col to message
    for(int i=0; i<n*b; i++){
      for(int j=0; j<n; j++){
        msgCol[N*(i%n) + n*(i/n) +j]= g[n*b + N*i +j];
      }
    }
    for(int i=n*(b+1); i<N; i++){
      for(int j=0; j<n; j++){
        msgCol[N*((i-n)%n)+n*((i-n)/n) +j]= g[n*b + N*i +j];
      }
    }
    // MPI bcast row+col
    mpi.bcast(msgRow.get(),n*(N-n));
    mpi.bcast(msgCol.get(),n*(N-n));
    for(auto&& rq: mpiAsyncRqTickets) mpi.waitRq(rq);   // Wait for all phase3 transfers to complete
    mpi.barrier();
  }
  /*
  printf("Launching GPU block algo with %d primary blocks\n", B);
  auto start= chrono::system_clock::now();
  d_g.copy(g.get(), N*N, Dir::H2D);
  for(int b=0; b<B; b++){
    phase1<n> <<<1,bs>>>(d_g, b*n, N);
    phase2<n> <<<dim3(B-1,2),bs>>>(d_g, b*n, b, N);
    phase3<n> <<<dim3(B-1,B-1),bs>>>(d_g, b*n, b, N);
  }
  d_g.copy(result_block.get(), N*N, Dir::D2H);
  auto GPUBlock_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("GPU block kernel done: %.3f\n", GPUBlock_time.count());
#ifdef LOG
  fprintf(logfile, "%.5f;", GPUBlock_time.count());
#endif
  auto check= test(result_block, groundTruth, N, "GPUblock");
  if(!check){
    printf("[GPUblock]: Test FAILED!\n");
    exit(1);
  }
  return GPUBlock_time; 
  */
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
int partitionRow(int rowStart, int rowSize, const int procN, int& startProc, int* remainingTilesProc, int* start, int* count){
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
}

// Partitions the section's tiles into processes. Output: start,count (for MPIscatterv)
// Returns the last covered tile in the section
int partitionSection(int sectionStart, const int row1Length, const int rowLength, const int rowsN, const int procN, int& lastRowReached, int* remainingTilesProc, int* start, int* count){
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
}

void scatterSection(MPIhandler& mpi, vector<int>& rqStore, int* g, int sectionStart, const int rowLength,
    const int rowN, int* tilesProc, int* counts, int* offsets){
  const int sectionSize= rowLength*rowN;
  int partEnd= 0, lastRowReached= 0;
  while(partEnd < sectionSize){      // Section not done
    partEnd+= partitionSection(sectionStart+ partEnd, (lastRowReached+1)*rowLength-partEnd, rowLength,
                               rowN-lastRowReached, mpi.procN(), lastRowReached, tilesProc, offsets, counts);
    rqStore.push_back(mpi.scatterTiles(g+sectionStart, counts, offsets));
  }
}


