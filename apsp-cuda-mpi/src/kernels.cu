#include "utils.h"  // Define MAX_THRperBLK, MAX_THRperBLK_MULTI macros

//################# Kernels ###############
// GPU kernels
#define n MAX_THRperBLK2D
//! g: submat of current process
//  tileStart: coordinate of first tile elt in g
//  pitch: size of each row of g
__global__ void phase1_krn(int* g, const xy tileStart, const int pitch){
  __shared__ int tile[n][n];
  // Load tile from global to shared
  tile[threadIdx.y][threadIdx.x]= g[ pitch*tileStart.y+tileStart.x+ pitch*threadIdx.y + threadIdx.x ];
  __syncthreads();

  /*if(!threadIdx.x&&!threadIdx.y){
    printf("[phase3krn]: \tb:(%d,%d) \ttStart=(%d,%d)\n\
        |%3d %3d %3d %3d\n\
        |%3d %3d %3d %3d\n\
        |%3d %3d %3d %3d\n\
        |%3d %3d %3d %3d\n", 
        blockIdx.x,blockIdx.y, tileStart.x,tileStart.y,
        tile[threadIdx.y][threadIdx.x], tile[threadIdx.y][threadIdx.x+1],
        tile[threadIdx.y][threadIdx.x+2], tile[threadIdx.y][threadIdx.x+3],
        tile[threadIdx.y+1][threadIdx.x], tile[threadIdx.y+1][threadIdx.x+1],
        tile[threadIdx.y+1][threadIdx.x+2], tile[threadIdx.y+1][threadIdx.x+3],
        tile[threadIdx.y+2][threadIdx.x], tile[threadIdx.y+2][threadIdx.x+1],
        tile[threadIdx.y+2][threadIdx.x+2], tile[threadIdx.y+2][threadIdx.x+3],
        tile[threadIdx.y+3][threadIdx.x], tile[threadIdx.y+3][threadIdx.x+1],
        tile[threadIdx.y+3][threadIdx.x+2], tile[threadIdx.y+3][threadIdx.x+3]
        );
  }*/

  // Calculate APSP in tile
  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > tile[threadIdx.y][k]+tile[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= tile[threadIdx.y][k]+tile[k][threadIdx.x];
    __syncthreads();
  }
  // Save to global
  g[ pitch*tileStart.y+tileStart.x+ pitch*threadIdx.y + threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

__global__ void phase2Row_krn(int* g, const int* primaryTile, const xy rstart, const int pitch, const int t_pitch){
  __shared__ int tile[n][n], primary[n][n];
  primary[threadIdx.y][threadIdx.x]= primaryTile[t_pitch*threadIdx.y + threadIdx.x];
  tile[threadIdx.y][threadIdx.x]= g[ pitch*rstart.y+ n*blockIdx.x+ pitch*threadIdx.y+threadIdx.x ];
  __syncthreads();

  /*if(!threadIdx.x&&!threadIdx.y){
    printf("[phase3krn]: \tb:(%d,%d) \tstart=(%d,%d)\n\
%3d %3d |%3d %3d \n\
%3d %3d |%3d %3d \n", 
        blockIdx.x,blockIdx.y, rstart.x,rstart.y,
        tile[threadIdx.y][threadIdx.x], tile[threadIdx.y][threadIdx.x+1],
        primary[threadIdx.y][threadIdx.x], primary[threadIdx.y][threadIdx.x+1],
        tile[threadIdx.y+1][threadIdx.x], tile[threadIdx.y+1][threadIdx.x+1],
        primary[threadIdx.y+1][threadIdx.x], primary[threadIdx.y+1][threadIdx.x+1]
        );
  }*/

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > tile[k][threadIdx.x]+primary[threadIdx.y][k])
      tile[threadIdx.y][threadIdx.x]= tile[k][threadIdx.x]+primary[threadIdx.y][k];
    __syncthreads();
  }

  g[ pitch*rstart.y+rstart.x+ n*blockIdx.x+ pitch*threadIdx.y+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

__global__ void phase2Col_krn(int* g, const int* primaryTile, const xy cstart, const int pitch, const int t_pitch){
  __shared__ int tile[n][n], primary[n][n];
  primary[threadIdx.y][threadIdx.x]= primaryTile[t_pitch*threadIdx.y + threadIdx.x];
  tile[threadIdx.y][threadIdx.x]= g[ cstart.x+ pitch*n*blockIdx.x+ pitch*threadIdx.y+threadIdx.x ];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > primary[k][threadIdx.x] + tile[threadIdx.y][k])
      tile[threadIdx.y][threadIdx.x]= primary[k][threadIdx.x] + tile[threadIdx.y][k];
    __syncthreads();
  }

  g[ pitch*cstart.y+cstart.x+ pitch*n*blockIdx.x+ pitch*threadIdx.y+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

// Actually calculate phase 3
__device__ void phase3_exec(int* g, const int* rowBuf, const int* colBuf, const int blockIdx_xskip,
    const int blockIdx_yskip, const int pitch, const int r_pitch, const int c_pitch){
  __shared__ int tile[n][n], row[n][n], col[n][n];
  row[threadIdx.y][threadIdx.x]= rowBuf[ n*blockIdx_xskip+ r_pitch*threadIdx.y+threadIdx.x ];
  col[threadIdx.y][threadIdx.x]= colBuf[ n*n*blockIdx_yskip+ n*threadIdx.y+threadIdx.x ];
  tile[threadIdx.y][threadIdx.x]= g[ pitch*n*blockIdx_yskip+ n*blockIdx_xskip+ pitch*threadIdx.y+threadIdx.x ];
  __syncthreads();

  
  /*if(!threadIdx.x&&!threadIdx.y){
    printf("[phase3krn]: \tb:(%d,%d) p=%3d rp=%3d cp=%3d\n\
        |%3d %3d\n\
        |%3d %3d\n\
         --------\n\
%3d %3d |%3d %3d\n\
%3d %3d |%3d %3d\n", 
        blockIdx.x,blockIdx.y, pitch, r_pitch, c_pitch,
        row[threadIdx.y][threadIdx.x],row[threadIdx.y][threadIdx.x+1],
        row[threadIdx.y+1][threadIdx.x],row[threadIdx.y+1][threadIdx.x+1],
        col[threadIdx.y][threadIdx.x], col[threadIdx.y][threadIdx.x+1],
        tile[threadIdx.y][threadIdx.x], tile[threadIdx.y][threadIdx.x+1],
        col[threadIdx.y+1][threadIdx.x], col[threadIdx.y+1][threadIdx.x+1],
        tile[threadIdx.y+1][threadIdx.x], tile[threadIdx.y+1][threadIdx.x+1]
        );
  }*/

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > col[threadIdx.y][k]+row[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= col[threadIdx.y][k]+row[k][threadIdx.x];
    __syncthreads();
  }

  g[ pitch*n*blockIdx_yskip+ n*blockIdx_xskip+ pitch*threadIdx.y+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}
// Determine if a row or column must be skipped and then call the actual calculation for phase 3
__global__ void phase3_krn(int* g, const int* rowBuf, const int* colBuf, const int b, const xy start,
    const int pitch, const int r_pitch, const int c_pitch){
  phase3_exec( g, rowBuf,colBuf, blockIdx.x, blockIdx.y, pitch, r_pitch,c_pitch);
}
__global__ void phase3r_krn(int* g, const int* rowBuf, const int* colBuf, const int b, const xy start,
    const int pitch, const int r_pitch, const int c_pitch){
  int blockIdx_yskip= (blockIdx.y >= b-start.y)? blockIdx.y+1: blockIdx.y;
  phase3_exec( g, rowBuf,colBuf, blockIdx.x, blockIdx_yskip, pitch, r_pitch,c_pitch);
}
__global__ void phase3c_krn(int* g, const int* rowBuf, const int* colBuf, const int b, const xy start,
    const int pitch, const int r_pitch, const int c_pitch){
  int blockIdx_xskip= (blockIdx.x >= b-start.x)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  phase3_exec( g, rowBuf,colBuf, blockIdx_xskip, blockIdx.y, pitch, r_pitch,c_pitch);
}
__global__ void phase3rc_krn(int* g, const int* rowBuf, const int* colBuf, const int b, const xy start,
    const int pitch, const int r_pitch, const int c_pitch){
  int blockIdx_xskip= (blockIdx.x >= b-start.x)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  int blockIdx_yskip= (blockIdx.y >= b-start.y)? blockIdx.y+1: blockIdx.y;
  phase3_exec( g, rowBuf,colBuf, blockIdx_xskip, blockIdx_yskip, pitch, r_pitch, c_pitch);
}
#undef n

//############# Kernel Wrappers ############
void phase1(const dim3 gs, const dim3 bs, int* g, const xy tileStart, const int pitch){
  phase1_krn<<<gs,bs>>>(g, tileStart, pitch);
}
void phase2Row(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const xy rowStart,
    const int pitch, const int t_pitch){
  phase2Row_krn<<<gs,bs>>>(g,primaryTile,rowStart,pitch,t_pitch);
}
void phase2Col(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const xy colStart,
    const int pitch, const int t_pitch){
  phase2Col_krn<<<gs,bs>>>(g,primaryTile,colStart,pitch,t_pitch);
}
void phase3(const dim3 gs, const dim3 bs, int* g, const int* row, const int* col, const int b, const xy start,
    const int pitch, const int r_pitch, const int c_pitch, const int rcFlag){
  switch(rcFlag){
    case 0:
      phase3_krn<<<gs,bs>>>( g,row,col, b,start, pitch, r_pitch,c_pitch);
      break;
    case 1:
      phase3r_krn<<<gs,bs>>>( g,row,col, b,start, pitch, r_pitch,c_pitch);
      break;
    case 2:
      phase3c_krn<<<gs,bs>>>( g,row,col, b,start, pitch, r_pitch,c_pitch);
      break;
    case 3:
      phase3rc_krn<<<gs,bs>>>( g,row,col, b,start, pitch, r_pitch,c_pitch);
  }
}

