#include <cstdio>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include "kernels.cu"
using namespace std;

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
#define MAX_THRpBLK2D 32
#define NO_TEST

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

enum Dir { H2D, D2H };

template<class T>
struct DPtr{
  DPtr(int N) { gpuErrchk(cudaMalloc(&data_, N*sizeof(T))); }
  ~DPtr() { cudaFree(data_); }
  void copy (T* a, int N, Dir dir) {
    if(dir == Dir::H2D) gpuErrchk(cudaMemcpy(data_, a, sizeof(T)*N, cudaMemcpyHostToDevice));
    else gpuErrchk(cudaMemcpy(a, data_, sizeof(T)*N, cudaMemcpyDeviceToHost));
  }
  T* get() const { return data_; }
  operator T*() const { return data_; }
  private:
  T* data_;
};

template<class T>
struct HPinPtr{
  HPinPtr(int N) { gpuErrchk(cudaHostAlloc(&data_, N*sizeof(T), cudaHostAllocDefault)); }
  ~HPinPtr() { cudaFreeHost(data_); }
  T& operator[](size_t i) const { return data_[i]; }
  operator T*() const { return data_; }
  T* get() const { return data_; } 
  private:
  T* data_;
};

void printG(unique_ptr<int[]>& g, int N){
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++)
      printf("%3d\t", g[i*N+j]);
    printf("\n");
  }
  printf("_____________________________________\n");
}

typedef chrono::duration<float, ratio<1>> Duration_fsec;

bool test(const HPinPtr<int>& toCheck, const unique_ptr<int[]>& truth, const int N, string name){
#ifndef NO_TEST
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      if(toCheck[i*N+j] != truth[i*N+j]){
        printf("[test/%s]: Error at (%d,%d)! toCheck/truth =\n\t%d\n\t%d\n", name.c_str(), i,j, toCheck[i*N+j],
            truth[i*N+j]);
        return false;
      }
#endif
  return true;
}

// CPU Floyd-Warshall
Duration_fsec run_cpu(const HPinPtr<int>& g, const int N, unique_ptr<int[]>& result_cpu){
  for(int i=0; i<N*N; i++) result_cpu[i]= g[i];     // Work on a copy of the data
  /*clk*/auto start= chrono::system_clock::now();
  for(int k=0; k<N; k++)
    for(int i=0; i<N; i++)
      for(int j=0; j<N; j++)
        if(result_cpu[i*N+j] > result_cpu[i*N+k]+result_cpu[k*N+j])
          result_cpu[i*N+j]= result_cpu[i*N+k]+result_cpu[k*N+j];
  /*clk*/auto cpu_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("CPU calc done: %1.3fs\n", cpu_time.count());
#ifdef LOG
  fprintf(logfile, "%1.3f;", cpu_time.count());
#endif
  return cpu_time;
}

// simple GPU Floyd-Warshall
Duration_fsec run_GPUsimple(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_simple(N*N);
  dim3 bs(MAX_THRpBLK2D, MAX_THRpBLK2D);
  if(N<MAX_THRpBLK2D) bs= dim3(N,N);
  dim3 gs(N/bs.x, N/bs.y);
  printf("Launching GPU simple algo...\n");
  /*clk*/auto start= chrono::system_clock::now();
  d_g.copy(g.get(), N*N, Dir::H2D);
  for(int k=0; k<N; k++) fw<<<gs,bs>>>(d_g, N, k);
  d_g.copy(result_simple.get(), N*N, Dir::D2H);
  /*clk*/auto GPUSimple_time= chrono::duration_cast<Duration_fsec>(chrono::system_clock::now() - start);
  printf("GPU simple kernel done: %.3f\n", GPUSimple_time.count());
#ifdef LOG
  fprintf(logfile, "%1.3f;", GPUSimple_time.count());
#endif
  auto check= test(result_simple, groundTruth, N, "GPUsimple");
  if(!check){
    printf("[GPUsimple]: Test FAILED!\n");
    exit(1);
  }
  return GPUSimple_time;
}

// GPU block algo
Duration_fsec run_GPUblock(const HPinPtr<int>& g, const int N, const unique_ptr<int[]>& groundTruth ){
  DPtr<int> d_g(N*N);
  cudaDeviceSynchronize();
  HPinPtr<int> result_block(N*N);
  constexpr const int n= MAX_THRpBLK2D;
  const int B= N/n;
  dim3 bs(MAX_THRpBLK2D, MAX_THRpBLK2D);
  if(N<MAX_THRpBLK2D) bs= dim3(N,N);
  dim3 gs(N/bs.x, N/bs.y);

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
  fprintf(logfile, "%.3f;", GPUBlock_time.count());
#endif
  auto check= test(result_block, groundTruth, N, "GPUblock");
  if(!check){
    printf("[GPUblock]: Test FAILED!\n");
    exit(1);
  }
  return GPUBlock_time; 
}

int main(int argc, char** argv){
  FILE* fin= stdin;
  if(argc>2 && !strcmp(argv[1],"-i")) fin= fopen(argv[2], "r");
  else if(argc>4 && strcmp(argv[3],"-i")) fin= fopen(argv[4], "r");
  if (fin==NULL){
    printf("Wrong input file\n");
    exit(3);
  }
#ifdef LOG
  if(argc<2 || (argv[1] != "-l" && (argc<4 || argv[3] != "-l"))){
    printf("Logging mode enabled. To run, specify logfile path as command line argument:\nUse: %s -l <logfile>\n", argv[0]);
    exit(2);
  }
  int l_idx= (argc==3)? 2: 4;
  FILE* logfile= fopen(argv[l_idx], "a");
  if(logfile==NULL){
    printf("Wrong log file\n");
    exit(4);
  }
  fprintf(logfile, "%d;", N);
#endif
#ifdef NO_TEST
  printf("WARNING! No_TEST has been defined\n\n");
#endif

  int N;
  while(!fscanf(fin, "%d\n", &N));
  HPinPtr<int> g(N*N);
  unique_ptr<int[]> groundTruth(new int[N*N]);
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      while(!fscanf(fin, "%d", &g[i*N+j]));
  printf("\nN=%d\n", N);
  
#ifndef NO_TEST
  run_cpu(g,N, groundTruth);
#endif
  run_GPUsimple(g,N, groundTruth);
  run_GPUblock(g,N, groundTruth);

#ifdef LOG
  fprintf(logfile, "\n");
  fclose(logfile);
#endif
  return 0;
}

