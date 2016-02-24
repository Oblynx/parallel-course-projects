#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <memory>
#include <string>

using namespace std;

#define INF INT_MAX

int main(int argc, char** argv){
  FILE* out;
  if (argc!=3 && argc!=4){
    printf("Use: %s <N> <p> <out_name>\n", argv[0]);
    return 1;
  }
  int N= atoi(argv[1]);
  float p= atof(argv[2]);
  out= (argc==4)? fopen(argv[3],"w"): stdout;

  unique_ptr<int[]> g(new int[N*N]);
  srand(time(NULL));
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++)
      g[i*N+j]= ((float)rand()/RAND_MAX < p)?
                  1 + rand()%(INF/N):
                  INF;
  for(int i=0; i<N; i++) g[i*N+i]= 0;

  fprintf(out, "%d\n", N);
  for(int i=0; i<N; i++){
    string line;
    for(int j=0; j<N-1; j++) line+= to_string(g[i*N+j])+"\t";
    line+= to_string(g[i*N+N-1])+"\n";
    fprintf(out, "%s", line.c_str());
  }
  fclose(out);
  return 0;
}
