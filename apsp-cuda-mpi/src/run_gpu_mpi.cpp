#include <cstdio>
#include <vector>
#include <ctime>
#include <cuda_runtime_api.h>
#include "utils.h"
#include "DPtr.h"
#include "mpi_handler.h"
#include "kernel_wrp.h"

using namespace std;

// Scatters the section that starts at sectionStart to all procs
void scatterSection(MPIhandler& mpi, vector<int>& rqStore, int* g, int sectionStart, const int rowLength,
    const int rowN, int* tilesProc, int* counts, int* offsets);

void copyPhase1(int* g, DPtr<int>& d_tile, const int b, const int n, const int N, int direction);
void copyPhase2(int* g, DPtr<int>& d_rowcol, const int b, const int n, const int N, int direction);
void copyRowcolMsg(int* g, int* msgRowcol, const int b, const int n, const int N);

// GPU block algo
double run_gpu_mpi_master(MPIhandler& mpi, int* g, int N, const int* groundTruth, FILE* logfile){
  // Constants
  const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(n,n);
  if(N<MAX_THRperBLK2D) bs= dim3(N,N);
  
  // MPI transfer setup
  mpi.splitMat(N);
  mpi.makeTypes(n,N);
  const int s_x= mpi.submatRowL(), s_y= mpi.submatRowN();

  // Allocate GPU memory
  DPtr<int> d_tile(n*n);        // 1st phase
  DPtr<int> d_rowcol(2*n*N);  // 2nd phase
  DPtr<int> d_submat(s_x*s_y);
  int* msgRowcol= new int[2*n*N];
  int* msgSubmat= new int[s_x*s_y];

  // Compute APSP
  // For every tile
  printf("Block size: %d\n", n);
  printf("Launching GPU block MPI algo with %d primary blocks\n", B);
  double begin= clock();

  // 1. Scatter submatrices
  mpi.scatterMat(g, msgSubmat);
  d_submat.copy(msgSubmat,s_x*s_y, 0);
  PRINTF("[run_gpu]: matrix scattered\n");
  //printG(g,N,n);
  for(int b=0; b<B; b++){
    //##### Compute Phase1&2 #####//
    copyPhase1(g,d_tile,b,n,N,0);          // Copy primary tile to GPU
    phase1(1,bs,d_tile);                        // Phase 1 kernel
    
    copyPhase2(g,d_rowcol,b,n,N,0);          // Copy row&col to GPU
    phase2(dim3(B-1,2),bs, d_rowcol,d_tile,b,N);        // Phase 2 kernel
    updateRowcol(2,bs, d_rowcol, d_tile, b,N);
    
    copyPhase2(g,d_rowcol,b,n,N,1);          // Copy row&col to CPU
    cudaDeviceSynchronize();
    PRINTF("[run_gpu]: b=%d phase2 complete\n",b);

    //##### MPI bcast tile, row, col #####//
    copyRowcolMsg(g,msgRowcol, b,n,N);         // Copy row&col to MPI 
    mpi.bcast(msgRowcol,2*n*N);
    mpi.barrier();
    PRINTF("[run_gpu]: b=%d row/col broadcasted\n",b);

    //##### Compute Phase3 #####//
    const int yStart= (mpi.submatStart()/n)/B, xStart= (mpi.submatStart()/n)%B, xEnd= B-xStart;
    updateSubmat(2*B-yStart-xStart,bs, d_submat, d_rowcol, b,N, xEnd,xStart,yStart,
                 mpi.submatRowL()/n,mpi.submatRowN()/n);
    phase3(dim3(s_x/n-1, s_y/n-1),bs, d_submat, d_rowcol, b,N, xStart,yStart, s_x);
    PRINTF("[run_gpu]: b=%d phase3 complete\n",b);

    d_submat.copy(msgSubmat,s_x*s_y, 1);
    mpi.gatherMat(msgSubmat,g);
    mpi.barrier();
    copyPhase2(g,d_rowcol,b,n,N,1);
    cudaDeviceSynchronize();
    //printG(g,N,n);
  }
  d_submat.copy(msgSubmat,s_x*s_y, 1);
  cudaDeviceSynchronize();
  mpi.gatherMat(msgSubmat,g);
  mpi.barrier();
  PRINTF("[run_gpu]: matrix gathered\n");
  double GPUBlock_time= (double)(clock() - begin) / CLOCKS_PER_SEC;
  printf("GPU block MPI algo done: %.3fsec\n", GPUBlock_time);

  //printf("Final G:\n"); printG(g,N,n);
  //printf("Truth:\n"); printG(groundTruth, N,n);

  #ifdef LOG
    fprintf(logfile, "%.5f;", GPUBlock_time);
  #endif
  bool check= test(g, groundTruth, N, "GPUblock_MPI");
  
  delete[](msgRowcol); delete[](msgSubmat);
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
  d_g.copy(g.get(), N*N, 0);
  for(int b=0; b<B; b++){
    phase1_multiy<n> <<<1,bs>>>(d_g, b*n, N);
    phase2_multiy<n> <<<dim3(B-1,2),bs>>>(d_g, b*n, b, N);
    phase3_multiy<n> <<<dim3(B-1,B-1),bs>>>(d_g, b*n, b, N);
  }
  d_g.copy(result_block.get(), N*N, 1);
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
// Copy primary tile to GPU
void copyPhase1(int* g, DPtr<int>& d_tile, const int b, const int n, const int N, int direction){
  for(int i=0; i<n; i++){
    d_tile.copy(g+ N*n*b+n*b +N*i,n, direction,n*i);
  }
}
// Copy current row&col to GPU
void copyPhase2(int*g, DPtr<int>& d_rowcol, const int b, const int n, const int N, int direction){
  d_rowcol.copy(g+ N*n*b, N*n, direction);              // Copy row
  for(int i=0; i<N; i++)                            // Copy col
    d_rowcol.copy(g+ n*b +N*i, n, direction, N*n+ N*(i%n)+ n*(i/n));
}
// Copy current row&col to msg
void copyRowcolMsg(int* g, int* msgRowcol, const int b, const int n, const int N){
  for(int j=0; j<n*N; j++) msgRowcol[j]= g[n*b*N +j];    // Row
  for(int i=0; i<N*n; i++)                               // Col
    msgRowcol[N*n+ (i%n)+ N*((i/n)%n)+ n*(i/(n*n))]= g[n*b+ (i%n)+ N*(i/n)];
}


