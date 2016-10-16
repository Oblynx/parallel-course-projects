#include "utils.h"  // Define MAX_THRperBLK, MAX_THRperBLK_MULTI macros

//################# Kernels ###############
// GPU kernels
#define n MAX_THRperBLK2D
//! g: submat of current process
//  tileStart: coordinate of first tile elt in g
//  pitch: size of each row of g
__global__ void phase1_krn(int* g, const int tileStart, const int pitch){
  __shared__ int tile[n][n];
  // Load tile from global to shared
  tile[threadIdx.y][threadIdx.x]= g[ tileStart+ pitch*threadIdx.y + threadIdx.x ];
  __syncthreads();
  // Calculate APSP in tile
  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > tile[threadIdx.y][k]+tile[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= tile[threadIdx.y][k]+tile[k][threadIdx.x];
    __syncthreads();
  }
  // Save to global
  g[ tileStart+ pitch*threadIdx.y + threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

__global__ void phase2Row_krn(int* g, const int* primaryTile, const int rowStart, const int pitch){
  __shared__ int tile[n][n], primary[n][n];
  primary[threadIdx.y][threadIdx.x]= primaryTile[n*threadIdx.y + threadIdx.x];
  tile[threadIdx.y][threadIdx.x]= g[ rowStart+ n*blockIdx.x+ pitch*threadIdx.y+threadIdx.x ];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > tile[k][threadIdx.x]+primary[threadIdx.y][k])
      tile[threadIdx.y][threadIdx.x]= tile[k][threadIdx.x]+primary[threadIdx.y][k];
    __syncthreads();
  }

  g[ rowStart+ n*blockIdx.x+ pitch*threadIdx.y+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

__global__ void phase2Col_krn(int* g, const int* primaryTile, const int colStart, const int pitch){
  __shared__ int tile[n][n], primary[n][n];
  primary[threadIdx.y][threadIdx.x]= primaryTile[n*threadIdx.y + threadIdx.x];
  tile[threadIdx.y][threadIdx.x]= g[ colStart+ pitch*n*blockIdx.x+ pitch*threadIdx.y+threadIdx.x ];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > primary[k][threadIdx.x] + tile[threadIdx.y][k])
      tile[threadIdx.y][threadIdx.x]= primary[k][threadIdx.x] + tile[threadIdx.y][k];
    __syncthreads();
  }

  g[ colStart+ pitch*n*blockIdx.x+ pitch*threadIdx.y+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

// Actually calculate phase 3
__device__ void phase3_exec(int* g, const int* rowBuf, const int* colBuf, const int blockIdx_xskip,
    const int blockIdx_yskip, const int pitch, const int rowpitch){
  __shared__ int tile[n][n], row[n][n], col[n][n];
  row[threadIdx.y][threadIdx.x]= rowBuf[ n*blockIdx_xskip+ rowpitch*threadIdx.y+threadIdx.x ];
  col[threadIdx.y][threadIdx.x]= colBuf[ n*n*blockIdx_yskip+ n*threadIdx.y+threadIdx.x ];
  tile[threadIdx.y][threadIdx.x]= g[ pitch*n*blockIdx_yskip+ n*blockIdx_xskip+ pitch*threadIdx.y+threadIdx.x ];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > col[threadIdx.y][k]+row[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= col[threadIdx.y][k]+row[k][threadIdx.x];
    __syncthreads();
  }

  g[ pitch*n*blockIdx_yskip+ n*blockIdx_xskip+ pitch*threadIdx.y+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}
// Determine if a row or column must be skipped and then call the actual calculation for phase 3
__global__ void phase3_krn(int* g, const int* rowBuf, const int* colBuf, const int b, const int xStart, const int
    yStart, const int pitch, const int rowpitch){
  phase3_exec( g, rowBuf,colBuf, blockIdx.x, blockIdx.y, pitch, rowpitch);
}
__global__ void phase3r_krn(int* g, const int* rowBuf, const int* colBuf, const int b, const int xStart, const int
    yStart, const int pitch, const int rowpitch){
  int blockIdx_yskip= (blockIdx.y >= b-yStart)? blockIdx.y+1: blockIdx.y;
  phase3_exec( g, rowBuf,colBuf, blockIdx.x, blockIdx_yskip, pitch, rowpitch);
}
__global__ void phase3c_krn(int* g, const int* rowBuf, const int* colBuf, const int b, const int xStart, const int
    yStart, const int pitch, const int rowpitch){
  int blockIdx_xskip= (blockIdx.x >= b-xStart)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  phase3_exec( g, rowBuf,colBuf, blockIdx_xskip, blockIdx.y, pitch, rowpitch);
}
__global__ void phase3rc_krn(int* g, const int* rowBuf, const int* colBuf, const int b, const int xStart, const int
    yStart, const int pitch, const int rowpitch){
  int blockIdx_xskip= (blockIdx.x >= b-xStart)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  int blockIdx_yskip= (blockIdx.y >= b-yStart)? blockIdx.y+1: blockIdx.y;
  phase3_exec( g, rowBuf,colBuf, blockIdx_xskip, blockIdx_yskip, pitch, rowpitch);
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
void phase1(const dim3 gs, const dim3 bs, int* g, const int tileStart, const int pitch){
  phase1_krn<<<gs,bs>>>(g, tileStart, pitch);
}
void phase2Row(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const int rowStart, const int pitch){
  phase2Row_krn<<<gs,bs>>>(g,primaryTile,rowStart,pitch);
}
void phase2Col(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const int colStart, const int pitch){
  phase2Col_krn<<<gs,bs>>>(g,primaryTile,colStart,pitch);
}
void phase3(const dim3 gs, const dim3 bs, int* g, const int* row, const int* col, const int b, const int xStart,
    const int yStart, const int pitch, const int rowpitch, const int rcFlag){
  switch(rcFlag){
    case 0:
      phase3_krn<<<gs,bs>>>( g,row,col, b,xStart,yStart, pitch, rowpitch);
      break;
    case 1:
      phase3r_krn<<<gs,bs>>>( g,row,col, b,xStart,yStart, pitch, rowpitch);
      break;
    case 2:
      phase3c_krn<<<gs,bs>>>( g,row,col, b,xStart,yStart, pitch, rowpitch);
      break;
    case 3:
      phase3rc_krn<<<gs,bs>>>( g,row,col, b,xStart,yStart, pitch, rowpitch);
  }
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
