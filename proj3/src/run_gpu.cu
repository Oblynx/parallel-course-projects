#include <cstdio>
#include <memory>
#include "utils.h"
#include "DPtr.cuh"
#include "kernels.cuh"

using namespace std;

#define MAX_THRpBLK2D 16
// simple GPU Floyd-Warshall
Duration_fsec run_GPUsimple(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_simple(N*N);
  dim3 bs(MAX_THRpBLK2D, MAX_THRpBLK2D);
  if(N<MAX_THRpBLK2D) bs= dim3(N,N);
  dim3 gs(N/bs.x, N/bs.y);
  printf("Launching GPU simple algo...\n");
  /*clk*/auto start= chrono::system_clock::now();
  d_g.copy(g.get(), N*N, Dir::H2D);
  for(int k=0; k<N; k++) fw_simple<<<gs,bs>>>(d_g, N, k);
  d_g.copy(result_simple.get(), N*N, Dir::D2H);
  /*clk*/auto GPUSimple_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("GPU simple kernel done: %.3f\n", GPUSimple_time.count());
#ifdef LOG
  fprintf(logfile, "%1.5f;", GPUSimple_time.count());
#endif
  auto check= test(result_simple, groundTruth, N, "GPUsimple");
  if(!check){
    printf("[GPUsimple]: Test FAILED!\n");
    exit(1);
  }
  return GPUSimple_time;
}
#undef MAX_THRpBLK2D

#define MAX_THRpBLK2D 16
// GPU block algo
Duration_fsec run_GPUblock(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile ){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_block(N*N);
  constexpr const int n= MAX_THRpBLK2D;
  const int B= N/n;
  dim3 bs(MAX_THRpBLK2D, MAX_THRpBLK2D);
  if(N<MAX_THRpBLK2D) bs= dim3(N,N);

  printf("Launching GPU block algo with %d primary blocks\n", B);
  /*clk*/auto start= chrono::system_clock::now();
  d_g.copy(g.get(), N*N, Dir::H2D);
  for(int b=0; b<B; b++){
    phase1<n> <<<1,bs>>>(d_g, b*n, N);
    phase2<n> <<<dim3(B-1,2),bs>>>(d_g, b*n, b, N);
    phase3<n> <<<dim3(B-1,B-1),bs>>>(d_g, b*n, b, N);
  }
  d_g.copy(result_block.get(), N*N, Dir::D2H);
  /*clk*/auto GPUBlock_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
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
}
#undef MAX_THRpBLK2D

#define MAX_THRpBLK2D 16
// GPU block algo -- multiple vertices per thread (x,y)
Duration_fsec run_GPUblock_multixy(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_block(N*N);
  constexpr const int n= MAX_THRpBLK2D;
  const int B= N/n;
  dim3 bs(n/2, n/2);
  if(N<MAX_THRpBLK2D) bs= dim3(N,N);

  printf("Launching GPU multi block algo with %d primary blocks\n", B);
  /*clk*/auto start= chrono::system_clock::now();
  d_g.copy(g.get(), N*N, Dir::H2D);
  for(int b=0; b<B; b++){
    phase1_multixy<n> <<<1,bs>>>(d_g, b*n, N);
    phase2_multixy<n> <<<dim3(B-1,2),bs>>>(d_g, b*n, b, N);
    phase3_multixy<n> <<<dim3(B-1,B-1),bs>>>(d_g, b*n, b, N);
  }
  d_g.copy(result_block.get(), N*N, Dir::D2H);
  /*clk*/auto GPUBlock_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("GPU multi block kernel done: %.3f\n", GPUBlock_time.count());
#ifdef LOG
  fprintf(logfile, "%.5f;", GPUBlock_time.count());
#endif
  auto check= test(result_block, groundTruth, N, "GPUblock_multi");
  if(!check){
    printf("[GPUblock_multi]: Test FAILED!\n");
    exit(1);
  }
  return GPUBlock_time; 
}
#undef MAX_THRpBLK2D

#define MAX_THRpBLK2D 16
// GPU block algo -- multiple vertices per thread (y only)
Duration_fsec run_GPUblock_multiy(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth,
    FILE* logfile ){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_block(N*N);
  constexpr const int n= MAX_THRpBLK2D;
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
#undef MAX_THRpBLK2D

