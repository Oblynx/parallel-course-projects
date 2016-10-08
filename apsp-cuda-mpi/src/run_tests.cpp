#include <cstdio>
#include "utils.h"
#include "DPtr.h"
#include "kernel_wrp.h"
using namespace std;

// CPU Floyd-Warshall
Duration_fsec run_cpu_test(const int* g, const int N, int* result_cpu, FILE* logfile){
  for(int i=0; i<N*N; i++) result_cpu[i]= g[i];     // Work on a copy of the data
  /*clk*/auto start= chrono::system_clock::now();
  for(int k=0; k<N; k++)
    for(int i=0; i<N; i++)
      for(int j=0; j<N; j++)
        if(result_cpu[i*N+j] > result_cpu[i*N+k]+result_cpu[k*N+j])
          result_cpu[i*N+j]= result_cpu[i*N+k]+result_cpu[k*N+j];
  /*clk*/auto cpu_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("--> CPU test done: %1.3fsec\n", cpu_time.count());
#ifdef LOG
  fprintf(logfile, "%1.3f;", cpu_time.count());
#endif
  return cpu_time;
}

// The GPU algorithm that was proven correct in the previous project
Duration_fsec run_gpu_test(const int* g, const int N, int* result_gpu, FILE* logfile){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_block(N*N);
  constexpr const int n= MAX_THRperBLK2D_MULTI;
  const int B= N/n;
  dim3 bs(n, n/2);

  printf("Launching GPU test (multiY block algo with %d primary blocks)\n", B);
  auto start= chrono::system_clock::now();
  d_g.copy(const_cast<int*>(g), N*N, Dir::H2D);
  for(int b=0; b<B; b++){
    phase1_multiy_test(1,bs, d_g, b*n,N);
    phase2_multiy_test(dim3(B-1,2),bs, d_g, b*n,b,N);
    phase3_multiy_test(dim3(B-1,B-1),bs, d_g, b*n,b,N);
  }
  d_g.copy(result_gpu, N*N, Dir::D2H);
  auto GPUBlock_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() -
      start);
  printf("--> GPU test done: %.3fsec\n", GPUBlock_time.count());
#ifdef LOG
  fprintf(logfile, "%.5f;", GPUBlock_time.count());
#endif
  return GPUBlock_time; 
}

