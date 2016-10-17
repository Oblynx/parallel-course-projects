#include <cstdio>
#include <ctime>
#include "utils.h"
#include "cuda_utils.h"
#include "kernel_wrp.h"
using namespace std;

// CPU Floyd-Warshall
double run_cpu_test(const int* g, const int N, int* result_cpu){
  for(int i=0; i<N*N; i++) result_cpu[i]= g[i];     // Work on a copy of the data
  printf("Launching CPU test\n");
  clock_t begin= clock();
  for(int k=0; k<N; k++)
    for(int i=0; i<N; i++)
      for(int j=0; j<N; j++)
        if(result_cpu[i*N+j] > result_cpu[i*N+k]+result_cpu[k*N+j])
          result_cpu[i*N+j]= result_cpu[i*N+k]+result_cpu[k*N+j];
  double cpu_time= (double)(clock() - begin) / CLOCKS_PER_SEC;
  printf("--> CPU test done: %1.3fsec\n", cpu_time);
  return cpu_time;
}

// The GPU algorithm that was proven correct in the previous project
double run_gpu_test(const int* g, const int N, int* result_gpu){
  HPinPtr<int> result_block(N*N);
  const int n= MAX_THRperBLK2D;
  const int B= N/n;
  dim3 bs(n, n);
  if(N<n) bs= dim3(N,N);
  DPtr<int> d_g(N*N);

  printf("Launching GPU test (block algo with %d primary blocks)\n", B);
  clock_t begin= clock();
  d_g.copyH2D(const_cast<int*>(g), N*N);
  for(int b=0; b<B; b++){
    phase1_test(1,bs, d_g, b*n,N);
    phase2_test(dim3(B-1,2),bs, d_g, b*n,b,N);
    phase3_test(dim3(B-1,B-1),bs, d_g, b*n,b,N);
  }
  d_g.copyD2H(result_gpu, N*N);
  double GPUBlock_time= (double)(clock() - begin) / CLOCKS_PER_SEC;
  printf("--> GPU test done: %.3fsec\n", GPUBlock_time);
  return GPUBlock_time; 
}
