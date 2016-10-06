#include <cstdio>
#include <memory>
#include <cuda_runtime_api.h>
#include "utils.h"
#include "DPtr.h"
#include "mpi_handler.h"
#include "kernel_wrp.h"

using namespace std;

// GPU block algo
Duration_fsec run_GPUblock(MPIhandler mpi, const int* g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile ){
  // Constants
  constexpr const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(MAX_THRperBLK2D, MAX_THRperBLK2D);
  if(N<MAX_THRperBLK2D) bs= dim3(N,N);
  
  // Allocate GPU memory for 1st,2nd phase
  DPtr<int> d_g1(n*n);    // 1st phase
  DPtr<int> d_g2(2*n*N);  // 1nd phase
  // For every tile
  for(int tile=0; tile<B; tile++){
    // Copy tile to GPU
    for(int i=0; i<n; i++){
      d_g1.copy(g+N*i,n,Dir::H2D,n*i);
    }
    // Phase 1 kernel
    phase1(1,bs,d_g1,N);
    // Copy row and column to GPU
    for(int i=0; i<; i++){
      d_g2.copy();
    }
    cudaStreamSynchonize(cudaStreamPerThread);
    // Phase 2 kernel
    ?;
    // Copy tile to CPU
    for(int i=0; i<n; i++){
      d_g1.copy(g+N*i,n,Dir::D2H,n*i);
    }
    cudaStreamSynchonize(cudaStreamPerThread);
    // Copy row and column to CPU
    ?;

    // MPI split phase 3

  }
  /*
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_block(N*N);
  constexpr const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(MAX_THRperBLK2D, MAX_THRperBLK2D);
  if(N<MAX_THRperBLK2D) bs= dim3(N,N);

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


// GPU block algo -- multiple vertices per thread (y only)
Duration_fsec run_GPUblock_multiy(MPIhandler mpi, const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile ){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_block(N*N);
  constexpr const int n= MAX_THRperBLK2D_MULTI;
  const int B= N/n;
  dim3 bs(n, n/2);

  printf("Launching GPU multi2 block algo with %d primary blocks\n", B);
  /*clk*/auto start= chrono::system_clock::now();
  d_g.copy(g.get(), N*N, Dir::H2D);
  for(int b=0; b<B; b++){
    phase1_multiy<n> <<<1,bs>>>(d_g, b*n, N);
    phase2_multiy<n> <<<dim3(B-1,2),bs>>>(d_g, b*n, b, N);
    phase3_multiy<n> <<<dim3(B-1,B-1),bs>>>(d_g, b*n, b, N);
  }
  d_g.copy(result_block.get(), N*N, Dir::D2H);
  /*clk*/auto GPUBlock_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
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

