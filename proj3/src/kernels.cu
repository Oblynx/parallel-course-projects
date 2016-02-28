__global__ void fw(int* g, int N, int k){
  int x= blockIdx.x*blockDim.x + threadIdx.x, y= blockIdx.y*blockDim.y + threadIdx.y;
  if(x<N && y<N){
    int v1= g[y*N+k], v2= g[k*N+x];
    if( g[y*N+x] > v1 + v2 ) g[y*N+x]= v1+v2;
  }
}

template<int n>
__global__ void phase1(int* g, const int pstart, const int N){
  __shared__ int tile[n][n];
  tile[threadIdx.y][threadIdx.x]= g[ (pstart+threadIdx.y)*N + pstart+threadIdx.x ];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > tile[threadIdx.y][k]+tile[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= tile[threadIdx.y][k]+tile[k][threadIdx.x];
    __syncthreads();
  }

  g[ (pstart+threadIdx.y)*N + pstart+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}

template<int n>
__global__ void phase2(int* g, const int pstart, const int primary_n, const int N){
  __shared__ int tile[n][n], primary[n][n];
  int blkIdx_skip= (blockIdx.x >= primary_n)? blockIdx.x+1: blockIdx.x;      // skip primary tile
  int x_t= (blockIdx.y)? blkIdx_skip*n+threadIdx.x: pstart+threadIdx.x;     // tile coordinates
  int y_t= (blockIdx.y)? pstart+threadIdx.y: blkIdx_skip*n+threadIdx.y;     // blkIdx,y? row: col
  primary[threadIdx.y][threadIdx.x]= g[ (pstart+threadIdx.y)*N + pstart+threadIdx.x ];
  tile[threadIdx.y][threadIdx.x]= g[y_t*N + x_t];
  __syncthreads();
  
  if(blockIdx.y)
    for(int k=0; k<n; k++){
      if(tile[threadIdx.y][threadIdx.x] > primary[threadIdx.y][k]+tile[k][threadIdx.x])
        tile[threadIdx.y][threadIdx.x]= primary[threadIdx.y][k]+tile[k][threadIdx.x];
      __syncthreads();
    }
  else
    for(int k=0; k<n; k++){
      if(tile[threadIdx.y][threadIdx.x] > tile[threadIdx.y][k]+primary[k][threadIdx.x])
        tile[threadIdx.y][threadIdx.x]= tile[threadIdx.y][k]+primary[k][threadIdx.x];
      __syncthreads();
    }

  g[y_t*N + x_t]= tile[threadIdx.y][threadIdx.x];
}

template<int n>
__global__ void phase3(int* g, const int pstart, const int primary_n, const int N){
  __shared__ int tile[n][n], row[n][n], col[n][n];
  int blkIdx_xskip= (blockIdx.x >= primary_n)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  int blkIdx_yskip= (blockIdx.y >= primary_n)? blockIdx.y+1: blockIdx.y;
  int x_t= blkIdx_xskip*n+threadIdx.x, y_t= blkIdx_yskip*n+threadIdx.y;     // tile coordinates
  row[threadIdx.y][threadIdx.x]= g[ (pstart+threadIdx.y)*N + x_t ];
  col[threadIdx.y][threadIdx.x]= g[ y_t*N + pstart+threadIdx.x   ];
  tile[threadIdx.y][threadIdx.x]= g[y_t*N + x_t];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[threadIdx.y][threadIdx.x] > col[threadIdx.y][k]+row[k][threadIdx.x])
      tile[threadIdx.y][threadIdx.x]= col[threadIdx.y][k]+row[k][threadIdx.x];
    __syncthreads();
  }

  g[y_t*N + x_t]= tile[threadIdx.y][threadIdx.x];
}

template<int n>
__global__ void copy1(int* g, const int pstart, const int N){
  __shared__ int tile[n][n];
  tile[threadIdx.y][threadIdx.x]= g[ (pstart+threadIdx.y)*N + pstart+threadIdx.x ];
  __syncthreads();

  tile[threadIdx.y][threadIdx.x]= -1;

  g[ (pstart+threadIdx.y)*N + pstart+threadIdx.x ]= tile[threadIdx.y][threadIdx.x];
}


template<int n>
__global__ void copy2(int* g, const int pstart, const int primary_n, const int N){
  __shared__ int tile[n][n], primary[n][n];
  int blkIdx_skip= (blockIdx.x >= primary_n)? blockIdx.x+1: blockIdx.x;      // skip primary tile
  int x_t= (blockIdx.y)? blkIdx_skip*n+threadIdx.x: pstart+threadIdx.x;     // tile coordinates
  int y_t= (blockIdx.y)? pstart+threadIdx.y: blkIdx_skip*n+threadIdx.y;     // blkIdx,y? row: col
  primary[threadIdx.y][threadIdx.x]= g[ (pstart+threadIdx.y)*N + pstart+threadIdx.x ];
  tile[threadIdx.y][threadIdx.x]= g[y_t*N + x_t];
  __syncthreads();


  tile[threadIdx.y][threadIdx.x]= -1;
  primary[threadIdx.y][threadIdx.x]= -2;
  g[y_t*N + x_t]= tile[threadIdx.y][threadIdx.x];
  g[ (pstart+threadIdx.y)*N + pstart+threadIdx.x ]= primary[threadIdx.y][threadIdx.x];
}

