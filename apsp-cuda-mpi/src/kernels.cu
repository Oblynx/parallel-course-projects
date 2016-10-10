#include "utils.h"  // Define MAX_THRperBLK, MAX_THRperBLK_MULTI macros

//################# Kernels ###############
// GPUblock kernels
#define n MAX_THRperBLK2D
__global__ void phase1_krn(int* g){
  __shared__ int tile[n][n];
  // Load tile from global to shared
  tile[threadIdx.y][threadIdx.x]= g[ threadIdx.y*n + threadIdx.x ];
  __syncthreads();
  // Calculate APSP in tile
  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > tile[threadIdx.y][k]+tile[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= tile[threadIdx.y][k]+tile[k][threadIdx.x];
    __syncthreads();
  }
  // Save to global
  g[ threadIdx.y*n + threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

__global__ void phase2_krn(int* g, const int* primaryTile, const int b, const int N){
  __shared__ int tile[n][n], primary[n][n];
  int blockIdx_xskip= (blockIdx.x >= b)? blockIdx.x+1: blockIdx.x;     // skip primary tile

  primary[threadIdx.y][threadIdx.x]= primaryTile[threadIdx.y*n + threadIdx.x];
  tile[threadIdx.y][threadIdx.x]= g[ blockIdx.y*N*n+ blockIdx_xskip*n + (threadIdx.y*N+threadIdx.x) ];
  __syncthreads();

  if(blockIdx.y)              // If column
    for(int k=0; k<n; k++){
      if(tile[threadIdx.y][threadIdx.x] > primary[k][threadIdx.x]+tile[threadIdx.y][k])
        tile[threadIdx.y][threadIdx.x]= primary[k][threadIdx.x]+tile[threadIdx.y][k];
      __syncthreads();
    }
  else                        // if row
    for(int k=0; k<n; k++){
      if(tile[threadIdx.y][threadIdx.x] > tile[k][threadIdx.x]+primary[threadIdx.y][k])
        tile[threadIdx.y][threadIdx.x]= tile[k][threadIdx.x]+primary[threadIdx.y][k];
      __syncthreads();
    }

  g[ blockIdx.y*N*n+ blockIdx_xskip*n + (threadIdx.y*N+threadIdx.x) ]= tile[threadIdx.y][threadIdx.x];
}

// Start: in blocks, not ints,relb= b-start 
__global__ void phase3_krn(int* g, const int* rowcol, const int b, const int N, const int xStart, const int yStart, const int rowL){
  __shared__ int tile[n][n], row[n][n], col[n][n];
  int blockIdx_xskip= (blockIdx.x >= b-xStart)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  int blockIdx_yskip= (blockIdx.y >= b-yStart)? blockIdx.y+1: blockIdx.y;
  int x_t= blockIdx_xskip*n+threadIdx.x, y_t= blockIdx_yskip*n+threadIdx.y;     // tile coordinates
  row[threadIdx.y][threadIdx.x]=  rowcol[ n*xStart+ threadIdx.y*N + x_t ];
  col[threadIdx.y][threadIdx.x]=  rowcol[ n*N+ n*yStart+ blockIdx_yskip*n+ threadIdx.y*N + threadIdx.x ];
  tile[threadIdx.y][threadIdx.x]= g[ y_t*rowL + x_t ];
  __syncthreads();


  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > col[threadIdx.y][k]+row[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= col[threadIdx.y][k]+row[k][threadIdx.x];
    __syncthreads();
  }

  g[ y_t*rowL + x_t ]= tile[threadIdx.y][threadIdx.x];
}
#undef n

// TODO
// GPUblock_multiy kernels
#define n MAX_THRperBLK2D_MULTI
__global__ void phase1_multiy_krn(int* g, const int pstart, const int N){
  __shared__ int tile[n][n];
  tile[2*threadIdx.y][threadIdx.x]= g[ (pstart+2*threadIdx.y)*N + pstart+threadIdx.x ];
  tile[2*threadIdx.y+1][threadIdx.x]= g[ (pstart+2*threadIdx.y+1)*N + pstart+threadIdx.x ];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[2*threadIdx.y][threadIdx.x] > tile[2*threadIdx.y][k]+tile[k][threadIdx.x])
      tile[2*threadIdx.y][threadIdx.x]= tile[2*threadIdx.y][k]+tile[k][threadIdx.x];
    if(tile[2*threadIdx.y+1][threadIdx.x] > tile[2*threadIdx.y+1][k]+tile[k][threadIdx.x])
      tile[2*threadIdx.y+1][threadIdx.x]= tile[2*threadIdx.y+1][k]+tile[k][threadIdx.x];
    __syncthreads();
  }

  g[ (pstart+2*threadIdx.y)*N + pstart+threadIdx.x ]= tile[2*threadIdx.y][threadIdx.x];
  g[ (pstart+2*threadIdx.y+1)*N + pstart+threadIdx.x ]= tile[2*threadIdx.y+1][threadIdx.x];
}

__global__ void phase2_multiy_krn(int* g, const int pstart, const int primary_n, const int N){
  __shared__ int tile[n][n], primary[n][n];
  int blkIdx_skip= (blockIdx.x >= primary_n)? blockIdx.x+1: blockIdx.x;      // skip primary tile
  int x_t= (blockIdx.y)? blkIdx_skip*n+threadIdx.x: pstart+threadIdx.x;     // tile coordinates
  int y_t= (blockIdx.y)? pstart+2*threadIdx.y: blkIdx_skip*n+2*threadIdx.y;     // blkIdx,y? row: col
  int y_t1= (blockIdx.y)? pstart+2*threadIdx.y+1: blkIdx_skip*n+2*threadIdx.y+1;     // blkIdx,y? row: col
  primary[2*threadIdx.y][threadIdx.x]= g[ (pstart+2*threadIdx.y)*N + pstart+threadIdx.x ];
  primary[2*threadIdx.y+1][threadIdx.x]= g[ (pstart+2*threadIdx.y+1)*N + pstart+threadIdx.x ];
  tile[2*threadIdx.y][threadIdx.x]= g[y_t*N + x_t];
  tile[2*threadIdx.y+1][threadIdx.x]= g[y_t1*N + x_t];
  __syncthreads();
  
  if(blockIdx.y)
    for(int k=0; k<n; k++){
      if(tile[2*threadIdx.y][threadIdx.x] > primary[2*threadIdx.y][k]+tile[k][threadIdx.x])
        tile[2*threadIdx.y][threadIdx.x]= primary[2*threadIdx.y][k]+tile[k][threadIdx.x];
      if(tile[2*threadIdx.y+1][threadIdx.x] > primary[2*threadIdx.y+1][k]+tile[k][threadIdx.x])
        tile[2*threadIdx.y+1][threadIdx.x]= primary[2*threadIdx.y+1][k]+tile[k][threadIdx.x];
      __syncthreads();
    }
  else
    for(int k=0; k<n; k++){
      if(tile[2*threadIdx.y][threadIdx.x] > tile[2*threadIdx.y][k]+primary[k][threadIdx.x])
        tile[2*threadIdx.y][threadIdx.x]= tile[2*threadIdx.y][k]+primary[k][threadIdx.x];
      if(tile[2*threadIdx.y+1][threadIdx.x] > tile[2*threadIdx.y+1][k]+primary[k][threadIdx.x])
        tile[2*threadIdx.y+1][threadIdx.x]= tile[2*threadIdx.y+1][k]+primary[k][threadIdx.x];
      __syncthreads();
    }

  g[y_t*N + x_t]= tile[2*threadIdx.y][threadIdx.x];
  g[y_t1*N + x_t]= tile[2*threadIdx.y+1][threadIdx.x];
}

__global__ void phase3_multiy_krn(int* g, const int pstart, const int primary_n, const int N){
  __shared__ int tile[n][n], row[n][n], col[n][n];
  int blkIdx_xskip= (blockIdx.x >= primary_n)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  int blkIdx_yskip= (blockIdx.y >= primary_n)? blockIdx.y+1: blockIdx.y;
  int x_t= blkIdx_xskip*n+threadIdx.x, y_t= blkIdx_yskip*n+2*threadIdx.y;     // tile coordinates
  int y_t1= blkIdx_yskip*n+2*threadIdx.y+1;     // tile coordinates
  row[2*threadIdx.y][threadIdx.x]= g[ (pstart+2*threadIdx.y)*N + x_t ];
  row[2*threadIdx.y+1][threadIdx.x]= g[ (pstart+2*threadIdx.y+1)*N + x_t ];
  col[2*threadIdx.y][threadIdx.x]= g[ y_t*N + pstart+threadIdx.x   ];
  col[2*threadIdx.y+1][threadIdx.x]= g[ y_t1*N + pstart+threadIdx.x   ];
  tile[2*threadIdx.y][threadIdx.x]= g[y_t*N + x_t];
  tile[2*threadIdx.y+1][threadIdx.x]= g[y_t1*N + x_t];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[2*threadIdx.y][threadIdx.x] > col[2*threadIdx.y][k]+row[k][threadIdx.x])
      tile[2*threadIdx.y][threadIdx.x]= col[2*threadIdx.y][k]+row[k][threadIdx.x];
    if(tile[2*threadIdx.y+1][threadIdx.x] > col[2*threadIdx.y+1][k]+row[k][threadIdx.x])
      tile[2*threadIdx.y+1][threadIdx.x]= col[2*threadIdx.y+1][k]+row[k][threadIdx.x];
    __syncthreads();
  }

  g[y_t*N + x_t]= tile[2*threadIdx.y][threadIdx.x];
  g[y_t1*N + x_t]= tile[2*threadIdx.y+1][threadIdx.x];
}
#undef n

//############# Kernel Wrappers ############
void phase1(const dim3 gs, const dim3 bs, int* g){
  phase1_krn<<<gs,bs>>>(g);
}
void phase2(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const int b, const int N){
  phase2_krn<<<gs,bs>>>(g,primaryTile,b,N);
}
void phase3(const dim3 gs, const dim3 bs, int* g, const int* rowcol, const int b, const int N, const int xStart, const int yStart, const int rowL){
  phase3_krn<<<gs,bs>>>(g,rowcol, b,N,xStart,yStart,rowL);
}

void phase1_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int N){
  phase1_multiy_krn<<<gs,bs>>>(g,pstart,N);
}
void phase2_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N){
  phase2_multiy_krn<<<gs,bs>>>(g,pstart,primary_n,N);
}
void phase3_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N){
  phase3_multiy_krn<<<gs,bs>>>(g,pstart,primary_n,N);
}

