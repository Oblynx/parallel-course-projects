#include <cstdio>
#include <memory>
#include <cuda_runtime.h>
#include "kernels.cu"
using namespace std;

#define MAX_THRpBLK2D 32

enum Dir { H2D, D2H };

template<class T>
struct DPtr{
  DPtr(int N) { cudaMalloc(&data_, N*N*sizeof(T)); }
  ~DPtr() { cudaFree(data_); }
  void copy(T* a, int N, Dir dir) {
    if(dir == Dir::H2D) cudaMemcpy(data_, a, sizeof(T)*N, cudaMemcpyHostToDevice);
    else cudaMemcpy(a, data_, sizeof(T)*N, cudaMemcpyDeviceToHost);
  }
  T* get() { return data_; }
  operator T*() { return data_; }
private:
  T* data_;
};

int main(){
  int N;
  scanf("%d\n", &N);
  unique_ptr<int[]> g(new int[N*N]);
  DPtr<int> d_g(N*N);
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      scanf("%d", &g[i*N+j]);

  printf("\nN=%d\n", N);
  for(int i=0; i<N;i++){
    for(int j=0;j<N;j++)
      printf("%3d ", g[i*N+j]);
    printf("\n");
  }
    printf("\n\n");

  // CPU Floyd-Warshall
  unique_ptr<int[]> result_cpu(new int[N*N]);
  for(int i=0; i<N*N; i++) result_cpu[i]= g[i];
  for(int k=0; k<N; k++)
    for(int i=0; i<N; i++)
      for(int j=0; j<N; j++)
        if(result_cpu[i*N+j] > result_cpu[i*N+k]+result_cpu[k*N+j])
          result_cpu[i*N+j]= result_cpu[i*N+k]+result_cpu[k*N+j];


  // simple GPU Floyd-Warshall
  d_g.copy(g.get(), N*N, Dir::H2D);
  dim3 bs(MAX_THRpBLK2D, MAX_THRpBLK2D);
  if(N<MAX_THRpBLK2D) bs= (N,N);
  dim3 gs(N/bs.x, N/bs.y);
  for(int k=0; k<N; k++) fw<<<gs,bs>>>(d_g, N, k);
  unique_ptr<int[]> result_simple(new int[N*N]);
  d_g.copy(result_simple.get(), N*N, Dir::D2H);

  // GPU block algo
  d_g.copy(g.get(), N*N, Dir::H2D);
  const int n= MAX_THRpBLK2D, B= N/n;
  for(int b=0; b< B; b++){
    phase1<MAX_THRpBLK2D> <<<1,bs>>>(d_g, b*n);
    phase2<MAX_THRpBLK2D> <<<(B-1,2),bs>>>(d_g, b*n, b);
    phase3<MAX_THRpBLK2D> <<<(B-1,B-1),bs>>>(d_g, b*n, b);
  }
  unique_ptr<int[]> result_block(new int[N*N]);
  d_g.copy(result_block.get(), N*N, Dir::D2H);

  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      if(result_simple[i*N+j] != result_cpu[i*N+j]){
        printf("[check]: Error at (%d,%d)! simple/cpu =\n\t%d\n\t%d\n", i,j, result_simple[i*N+j],
            result_cpu[i*N+j]);
        exit(1);
      }
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      if(result_block[i*N+j] != result_cpu[i*N+j]){
        printf("[check]: Error at (%d,%d)! block/cpu =\n\t%d\n\t%d\n", i,j, result_block[i*N+j],
            result_cpu[i*N+j]);
        exit(1);
      }

  return 0;
}

