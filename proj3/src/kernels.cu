__global__ void fw(int* g, int N, int k){
  int x= blockIdx.x*blockDim.x + threadIdx.x, y= blockIdx.y*blockDim.y + threadIdx.y;
  int v1= g[y*N+k], v2= g[k*N+x];
  if( g[y*N+x] > v1 + v2 ) g[y*N+x]= v1+v2;
}

#define n (blockDim.x)

__global__ void phase1(int* g, int pstart){
  __shared__ int tile[n][n];
  tile[threadIdx.y][threadIdx.x]= g[ (pstart+threadIdx.y)*n + pstart+threadIdx.x ];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > tile[threadIdx.y][k]+tile[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= tile[threadIdx.y][k]+tile[k][threadIdx.x];
    __syncthreads();
  }

  g[ (pstart+threadIdx.y)*n + pstart+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

__global__ void phase2(int* g, int pstart, int primary_n){
  __shared__ int tile[n][n], pr[n][n];
  int blkIdx_skip= (blockIdx.x > primary_n)? blockIdx.x+1: blockIdx.x;      // skip primary tile
  int x_t= (blockIdx.y)? blkIdx_skip*n+threadIdx.x: pstart+threadIdx.x;     // tile coordinates
  int y_t= (blockIdx.y)? pstart+threadIdx.y: blkIdx_skip*n+threadIdx.y;     // blkIdx,y? row: col
  primary[threadIdx.y][threadIdx.x]= g[ (pstart+threadIdx.y)*n + pstart+threadIdx.x ];
  tile[threadIdx.y][threadIdx.x]= g[y_t*n + x_t];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > primary[threadIdx.y][k]+primary[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= primary[threadIdx.y][k]+primary[k][threadIdx.x];
    __syncthreads();
  }

  g[y_t*n + x_t]= tile[threadIdx.y][threadIdx.x];
}

__global__ void phase3(int* g, int pstart, int primary_n){
  __shared__ int tile[n][n], row[n][n], col[n][n];
  int blkIdx_xskip= (blockIdx.x > primary_n)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  int blkIdx_yskip= (blockIdx.y > primary_n)? blockIdx.y+1: blockIdx.y;
  int x_t= blkIdx_xskip*n+threadIdx.x, y_t= blkIdx_yskip*n+threadIdx.y;     // tile coordinates
  row[threadIdx.y][threadIdx.x]= g[ (pstart+threadIdx.y)*n + x_t ];
  col[threadIdx.y][threadIdx.x]= g[ y_t*n + pstart+threadIdx.x   ];
  tile[threadIdx.y][threadIdx.x]= g[y_t*n + x_t];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > col[threadIdx.y][k]+row[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= col[threadIdx.y][k]+row[k][threadIdx.x];
    __syncthreads();
  }

  g[y_t*n + x_t]= tile[threadIdx.y][threadIdx.x];
}
