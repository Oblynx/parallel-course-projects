#include "utils.h"
using namespace std;

// CPU Floyd-Warshall
Duration_fsec run_cpu(const HPinPtr<int>& g, const int N, unique_ptr<int[]>& result_cpu, FILE* logfile){
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

