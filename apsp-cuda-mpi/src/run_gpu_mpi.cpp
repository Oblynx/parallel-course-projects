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
Duration_fsec run_gpu_mpi_master(MPIhandler& mpi, int* g, int N, const int* groundTruth, FILE* logfile){
  // Constants
  constexpr const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(n,n);
  if(N<MAX_THRperBLK2D) bs= dim3(N,N);
  
  // MPI transfer setup
  mpi.splitMat(N);
  mpi.makeTypes(n,N);
  const int s_x= mpi.submatRowL(), s_y= mpi.submatRowN();

  // Allocate GPU memory
  DPtr<int> d_g1(n*n);        // 1st phase
  DPtr<int> d_g2(2*n*N);  // 2nd phase
  DPtr<int> d_g3(s_x*s_y);
  unique_ptr<int[]> msgRowcol(new int[2*n*N]);
  unique_ptr<int[]> msgSubmat(new int[s_x*s_y]);

  // Compute APSP
  // For every tile
  printf("Block size: %d\n", n);
  printf("Launching GPU block MPI algo with %d primary blocks\n", B);
  auto start= chrono::system_clock::now();
  for(int b=0; b<B; b++){
    printG(g,N,n);

    /*
      copyPhase1(g,d_g1,b,n,N,Dir::H2D);          // Copy primary tile to GPU
      phase1(1,bs,d_g1);                        // Phase 1 kernel
      PRINTF("[run_gpu]: ph1 launched\n");
      copyPhase1(g,d_g1,b,n,N,Dir::D2H);          // Copy primary tile to CPU
      cudaStreamSynchronize(cudaStreamPerThread); // Finish phase1, row,col copies
      PRINTF("[run_gpu]: b=%d phase1 complete\n",b);
      printG(g,N,n);
      
      copyPhase2(g,d_g2,b,n,N,Dir::H2D);          // Copy row&col to GPU
      phase2(dim3(B-1,2),bs, d_g2,d_g1,b,N);      // Phase 2 kernel
      PRINTF("[run_gpu]: ph2 launched\n");
      copyPhase2(g,d_g2,b,n,N,Dir::D2H);          // Copy row&col to CPU
      cudaStreamSynchronize(cudaStreamPerThread); // Finish phase2,tile
      PRINTF("[run_gpu]: b=%d phase2 complete\n",b);
      printG(g,N,n);

      d_g3.copy(msgSubmat.get(),s_x*s_y, Dir::H2D);
      const int yStart= (mpi.submatStart()/n)/B, xStart= (mpi.submatStart()/n)%B;
      phase3(dim3(s_x/n, s_y/n),bs, d_g3, d_g2, b,N, xStart,yStart, s_x);
      PRINTF("[run_gpu]: ph3 launched\n");
      d_g3.copy(msgSubmat.get(),s_x*s_y, Dir::D2H);
      cudaStreamSynchronize(cudaStreamPerThread);
      PRINTF("[run_gpu]: b=%d phase3 complete\n",b);
      printG(g,N,n);
    */

    // MPI split phase 3
    //vector<int> mpiAsyncRqTickets(4);
    mpi.scatterMat(g, msgSubmat.get());
    PRINTF("[run_gpu]: b=%d matrix scattered\n",b);

    //##### Compute Phase1&2 #####//
    copyPhase1(g,d_g1,b,n,N,Dir::H2D);          // Copy primary tile to GPU
    phase1(1,bs,d_g1);                        // Phase 1 kernel
    PRINTF("[run_gpu]: ph1 launched\n");
    
    copyPhase2(g,d_g2,b,n,N,Dir::H2D);          // Copy row&col to GPU
    cudaStreamSynchronize(cudaStreamPerThread); // Finish phase1, row,col copies
    PRINTF("[run_gpu]: b=%d phase1 complete\n",b);
    phase2(dim3(B-1,2),bs, d_g2,d_g1,b,N);        // Phase 2 kernel
    
    copyPhase1(g,d_g1,b,n,N,Dir::D2H);          // Copy primary tile to CPU
    cudaStreamSynchronize(cudaStreamPerThread); // Finish phase2,tile

    copyPhase2(g,d_g2,b,n,N,Dir::D2H);          // Copy row&col to CPU
    cudaStreamSynchronize(cudaStreamPerThread);
    PRINTF("[run_gpu]: b=%d phase2 complete\n",b);
    printG(g,N,n);
    // Phases 1&2 + CPU/GPU transfers complete
    
    //##### MPI tile, row, col #####//
    copyRowcolMsg(g,msgRowcol.get(), b,n,N);         // Copy row&col to CPU
    // MPI bcast row+col
    mpi.bcast(msgRowcol.get(),2*n*N);
    PRINTF("[run_gpu]: b=%d row/col broadcasted\n",b);
    //for(auto&& rq: mpiAsyncRqTickets) mpi.waitRq(rq);   // Wait for all phase3 transfers to complete

    //##### Compute Phase3 #####//
    d_g3.copy(msgSubmat.get(),s_x*s_y, Dir::H2D);
    const int yStart= (mpi.submatStart()/n)/B, xStart= (mpi.submatStart()/n)%B;
    phase3(dim3(s_x/n-1, s_y/n-1),bs, d_g3, d_g2, b,N, xStart,yStart, s_x);
    d_g3.copy(msgSubmat.get(),s_x*s_y, Dir::D2H);
    cudaStreamSynchronize(cudaStreamPerThread);
    PRINTF("[run_gpu]: b=%d phase3 complete\n",b);

    PRINTF("[run_gpu]: MPI submat:\n");
    for(int i=0; i<s_y; i++){
      for(int j=0; j<s_x; j++){
        printf("%3d ", msgSubmat[i*s_x+j]);
      }
      printf("\n");
    }
    printf("\n");

    mpi.gatherMat(msgSubmat.get(),g);
    mpi.barrier();
    PRINTF("[run_gpu]: b=%d matrix gathered\n",b);
    printG(g,N,n);
  }
  auto GPUBlock_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("GPU block kernel done: %.3fsec\n", GPUBlock_time.count());

  printG(groundTruth, N,n);

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
// Copy primary tile to GPU
void copyPhase1(int* g, DPtr<int>& d_g1, const int b, const int n, const int N, Dir direction){
  for(int i=0; i<n; i++){
    d_g1.copy(g+ N*n*b+n*b +N*i,n, direction,n*i);
  }
}
// Copy current row&col to GPU
void copyPhase2(int*g, DPtr<int>& d_g2, const int b, const int n, const int N, Dir direction){
  d_g2.copy(g+ N*n*b, N*n, direction);              // Copy row
  for(int i=0; i<N; i++)                            // Copy col
    d_g2.copy(g+ n*b +N*i, n, direction, N*n+ N*(i%n)+ n*(i/n));
}
// Copy current row&col to msg
void copyRowcolMsg(int* g, int* msgRowcol, const int b, const int n, const int N){
  for(int j=0; j<n*N; j++) msgRowcol[j]= g[n*b*N +j];    // Row
  for(int i=0; i<N*n; i++)                               // Col
    msgRowcol[N*n+ (i%n)+ N*((i/n)%n)+ n*(i/(n*n))]= g[n*b+ (i%n)+ N*(i/n)];
}


