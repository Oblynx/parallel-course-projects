#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <memory>
#include <string>
#include <thread>

using namespace std;

#define INF 1<<29

int main(int argc, char** argv){
  FILE* out;
  if (argc!=3 && argc!=4){
    printf("Use: %s <N> <p> [<out_name>]\n", argv[0]);
    return 1;
  }
  const int logN= atoi(argv[1]), N= 1<<logN, mod= (INF/N)? INF/N: 1;
  const float p= atof(argv[2]);
  out= (argc==4)? fopen(argv[3],"w"): stdout;

  unique_ptr<int[]> g(new int[N*N]);
  printf("[makeGraph]: Starting gen logN=%d\n", logN);
  
  unsigned seed= time(NULL);
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      g[i*N+j]= ((float)rand_r(&seed)/RAND_MAX < p)?
                  1 + rand_r(&seed)%mod:
                  INF;
    }
  }
  for(int i=0; i<N; i++) g[i*N+i]= 0;

  fprintf(out, "%d\n", logN);
  for(int i=0; i<N; i++){
    string line;
    for(int j=0; j<N-1; j++) line+= to_string(g[i*N+j])+"\t";
    line+= to_string(g[i*N+N-1])+"\n";
    fprintf(out, "%s", line.c_str());
  }
  fclose(out);
  return 0;
}
