#include <cstdio>
#include <memory>
#include <cuda_runtime.h>
#include "kernels.cu"
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define MAX_THRpBLK2D 32

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

enum Dir { H2D, D2H };

template<class T>
struct DPtr{
  DPtr(int N) { gpuErrchk(cudaMalloc(&data_, N*sizeof(T))) }
  ~DPtr() { cudaFree(data_); }
  void copy(T* a, int N, Dir dir) {
    if(dir == Dir::H2D) gpuErrchk(cudaMemcpy(data_, a, sizeof(T)*N, cudaMemcpyHostToDevice))
    else gpuErrchk(cudaMemcpy(a, data_, sizeof(T)*N, cudaMemcpyDeviceToHost))
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
/*
  for(int i=0; i<N;i++){
    for(int j=0;j<N;j++)
      printf("%3d ", g[i*N+j]);
    printf("\n");
  }
    printf("\n\n");
*/
  // CPU Floyd-Warshall
  unique_ptr<int[]> result_cpu(new int[N*N]);
  for(int i=0; i<N*N; i++) result_cpu[i]= g[i];
  for(int k=0; k<N; k++)
    for(int i=0; i<N; i++)
      for(int j=0; j<N; j++)
        if(result_cpu[i*N+j] > result_cpu[i*N+k]+result_cpu[k*N+j])
          result_cpu[i*N+j]= result_cpu[i*N+k]+result_cpu[k*N+j];
printf("CPU calc done\n");

  // simple GPU Floyd-Warshall
  d_g.copy(g.get(), N*N, Dir::H2D);
  dim3 bs(MAX_THRpBLK2D, MAX_THRpBLK2D);
  if(N<MAX_THRpBLK2D) bs= (N,N);
  dim3 gs(N/bs.x, N/bs.y);
printf("Launching simple kernel...\n");
  for(int k=0; k<N; k++) fw<<<gs,bs>>>(d_g, N, k);
  unique_ptr<int[]> result_simple(new int[N*N]);
  d_g.copy(result_simple.get(), N*N, Dir::D2H);
printf("GPU simple kernel done\n");

  // GPU block algo
  d_g.copy(g.get(), N*N, Dir::H2D);
  constexpr const int n= MAX_THRpBLK2D;
  const int B= N/n;
printf("Launching GPU block kernel, B=%d\n", B);
  for(int b=0; b<B; b++){
    phase1<n> <<<1,bs>>>(d_g, b*n);
    phase2<n> <<<(B-1,2),bs>>>(d_g, b*n, b);
    phase3<n> <<<(B-1,B-1),bs>>>(d_g, b*n, b);
  }
  unique_ptr<int[]> result_block(new int[N*N]);
  d_g.copy(result_block.get(), N*N, Dir::D2H);
printf("GPU block kernel done\n");

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

