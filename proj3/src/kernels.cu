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
__global__ void phase1m(int* g, const int pstart, const int N){
  __shared__ int tile[n][n];
  tile[2*threadIdx.y][2*threadIdx.x]= g[ (pstart+2*threadIdx.y)*N + pstart+2*threadIdx.x ];
  tile[2*threadIdx.y][2*threadIdx.x+1]= g[ (pstart+2*threadIdx.y)*N + pstart+2*threadIdx.x+1 ];
  tile[2*threadIdx.y+1][2*threadIdx.x]= g[ (pstart+2*threadIdx.y+1)*N + pstart+2*threadIdx.x ];
  tile[2*threadIdx.y+1][2*threadIdx.x+1]= g[ (pstart+2*threadIdx.y+1)*N + pstart+2*threadIdx.x+1 ];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[2*threadIdx.y][2*threadIdx.x] > tile[2*threadIdx.y][k]+tile[k][2*threadIdx.x])
      tile[2*threadIdx.y][2*threadIdx.x]= tile[2*threadIdx.y][k]+tile[k][2*threadIdx.x];
    if(tile[2*threadIdx.y][2*threadIdx.x+1] > tile[2*threadIdx.y][k]+tile[k][2*threadIdx.x+1])
      tile[2*threadIdx.y][2*threadIdx.x+1]= tile[2*threadIdx.y][k]+tile[k][2*threadIdx.x+1];
    if(tile[2*threadIdx.y+1][2*threadIdx.x] > tile[2*threadIdx.y+1][k]+tile[k][2*threadIdx.x])
      tile[2*threadIdx.y+1][2*threadIdx.x]= tile[2*threadIdx.y+1][k]+tile[k][2*threadIdx.x];
    if(tile[2*threadIdx.y+1][2*threadIdx.x+1] > tile[2*threadIdx.y+1][k]+tile[k][2*threadIdx.x+1])
      tile[2*threadIdx.y+1][2*threadIdx.x+1]= tile[2*threadIdx.y+1][k]+tile[k][2*threadIdx.x+1];
    __syncthreads();
  }

  g[ (pstart+2*threadIdx.y)*N + pstart+2*threadIdx.x ]= tile[2*threadIdx.y][2*threadIdx.x];
  g[ (pstart+2*threadIdx.y)*N + pstart+2*threadIdx.x+1 ]= tile[2*threadIdx.y][2*threadIdx.x+1];
  g[ (pstart+2*threadIdx.y+1)*N + pstart+2*threadIdx.x ]= tile[2*threadIdx.y+1][2*threadIdx.x];
  g[ (pstart+2*threadIdx.y+1)*N + pstart+2*threadIdx.x+1 ]= tile[2*threadIdx.y+1][2*threadIdx.x+1];
}

template<int n>
__global__ void phase2m(int* g, const int pstart, const int primary_n, const int N){
  __shared__ int tile[n][n], primary[n][n];
  int blkIdx_skip= (blockIdx.x >= primary_n)? blockIdx.x+1: blockIdx.x;      // skip primary tile
  int x_t= (blockIdx.y)? blkIdx_skip*n+2*threadIdx.x: pstart+2*threadIdx.x;     // tile coordinates
  int x_t1= (blockIdx.y)? blkIdx_skip*n+2*threadIdx.x+1: pstart+2*threadIdx.x+1;     // tile coordinates
  int y_t= (blockIdx.y)? pstart+2*threadIdx.y: blkIdx_skip*n+2*threadIdx.y;     // blkIdx,y? row: col
  int y_t1= (blockIdx.y)? pstart+2*threadIdx.y+1: blkIdx_skip*n+2*threadIdx.y+1;     // blkIdx,y? row: col
  primary[2*threadIdx.y][2*threadIdx.x]= g[ (pstart+2*threadIdx.y)*N + pstart+2*threadIdx.x ];
  primary[2*threadIdx.y][2*threadIdx.x+1]= g[ (pstart+2*threadIdx.y)*N + pstart+2*threadIdx.x+1 ];
  primary[2*threadIdx.y+1][2*threadIdx.x]= g[ (pstart+2*threadIdx.y+1)*N + pstart+2*threadIdx.x ];
  primary[2*threadIdx.y+1][2*threadIdx.x+1]= g[ (pstart+2*threadIdx.y+1)*N + pstart+2*threadIdx.x+1 ];
  tile[2*threadIdx.y][2*threadIdx.x]= g[y_t*N + x_t];
  tile[2*threadIdx.y][2*threadIdx.x+1]= g[y_t*N + x_t1];
  tile[2*threadIdx.y+1][2*threadIdx.x]= g[y_t1*N + x_t];
  tile[2*threadIdx.y+1][2*threadIdx.x+1]= g[y_t1*N + x_t1];
  __syncthreads();
  
  if(blockIdx.y)
    for(int k=0; k<n; k++){
      if(tile[2*threadIdx.y][2*threadIdx.x] > primary[2*threadIdx.y][k]+tile[k][2*threadIdx.x])
        tile[2*threadIdx.y][2*threadIdx.x]= primary[2*threadIdx.y][k]+tile[k][2*threadIdx.x];
      if(tile[2*threadIdx.y][2*threadIdx.x+1] > primary[2*threadIdx.y][k]+tile[k][2*threadIdx.x+1])
        tile[2*threadIdx.y][2*threadIdx.x+1]= primary[2*threadIdx.y][k]+tile[k][2*threadIdx.x+1];
      if(tile[2*threadIdx.y+1][2*threadIdx.x] > primary[2*threadIdx.y+1][k]+tile[k][2*threadIdx.x])
        tile[2*threadIdx.y+1][2*threadIdx.x]= primary[2*threadIdx.y+1][k]+tile[k][2*threadIdx.x];
      if(tile[2*threadIdx.y+1][2*threadIdx.x+1] > primary[2*threadIdx.y+1][k]+tile[k][2*threadIdx.x+1])
        tile[2*threadIdx.y+1][2*threadIdx.x+1]= primary[2*threadIdx.y+1][k]+tile[k][2*threadIdx.x+1];
      __syncthreads();
    }
  else
    for(int k=0; k<n; k++){
      if(tile[2*threadIdx.y][2*threadIdx.x] > tile[2*threadIdx.y][k]+primary[k][2*threadIdx.x])
        tile[2*threadIdx.y][2*threadIdx.x]= tile[2*threadIdx.y][k]+primary[k][2*threadIdx.x];
      if(tile[2*threadIdx.y][2*threadIdx.x+1] > tile[2*threadIdx.y][k]+primary[k][2*threadIdx.x+1])
        tile[2*threadIdx.y][2*threadIdx.x+1]= tile[2*threadIdx.y][k]+primary[k][2*threadIdx.x+1];
      if(tile[2*threadIdx.y+1][2*threadIdx.x] > tile[2*threadIdx.y+1][k]+primary[k][2*threadIdx.x])
        tile[2*threadIdx.y+1][2*threadIdx.x]= tile[2*threadIdx.y+1][k]+primary[k][2*threadIdx.x];
      if(tile[2*threadIdx.y+1][2*threadIdx.x+1] > tile[2*threadIdx.y+1][k]+primary[k][2*threadIdx.x+1])
        tile[2*threadIdx.y+1][2*threadIdx.x+1]= tile[2*threadIdx.y+1][k]+primary[k][2*threadIdx.x+1];
      __syncthreads();
    }

  g[y_t*N + x_t]= tile[2*threadIdx.y][2*threadIdx.x];
  g[y_t*N + x_t1]= tile[2*threadIdx.y][2*threadIdx.x+1];
  g[y_t1*N + x_t]= tile[2*threadIdx.y+1][2*threadIdx.x];
  g[y_t1*N + x_t1]= tile[2*threadIdx.y+1][2*threadIdx.x+1];
}

template<int n>
__global__ void phase3m(int* g, const int pstart, const int primary_n, const int N){
  __shared__ int tile[n][n], row[n][n], col[n][n];
  int blkIdx_xskip= (blockIdx.x >= primary_n)? blockIdx.x+1: blockIdx.x;     // skip primary tile
  int blkIdx_yskip= (blockIdx.y >= primary_n)? blockIdx.y+1: blockIdx.y;
  int x_t= blkIdx_xskip*n+2*threadIdx.x, y_t= blkIdx_yskip*n+2*threadIdx.y;     // tile coordinates
  int x_t1= blkIdx_xskip*n+2*threadIdx.x+1, y_t1= blkIdx_yskip*n+2*threadIdx.y+1;     // tile coordinates
  row[2*threadIdx.y][2*threadIdx.x]= g[ (pstart+2*threadIdx.y)*N + x_t ];
  row[2*threadIdx.y][2*threadIdx.x+1]= g[ (pstart+2*threadIdx.y)*N + x_t1 ];
  row[2*threadIdx.y+1][2*threadIdx.x]= g[ (pstart+2*threadIdx.y+1)*N + x_t ];
  row[2*threadIdx.y+1][2*threadIdx.x+1]= g[ (pstart+2*threadIdx.y+1)*N + x_t1 ];
  col[2*threadIdx.y][2*threadIdx.x]= g[ y_t*N + pstart+2*threadIdx.x   ];
  col[2*threadIdx.y][2*threadIdx.x+1]= g[ y_t*N + pstart+2*threadIdx.x+1   ];
  col[2*threadIdx.y+1][2*threadIdx.x]= g[ y_t1*N + pstart+2*threadIdx.x   ];
  col[2*threadIdx.y+1][2*threadIdx.x+1]= g[ y_t1*N + pstart+2*threadIdx.x+1   ];
  tile[2*threadIdx.y][2*threadIdx.x]= g[y_t*N + x_t];
  tile[2*threadIdx.y][2*threadIdx.x+1]= g[y_t*N + x_t1];
  tile[2*threadIdx.y+1][2*threadIdx.x]= g[y_t1*N + x_t];
  tile[2*threadIdx.y+1][2*threadIdx.x+1]= g[y_t1*N + x_t1];
  __syncthreads();

  for(int k=0; k<n; k++){
    if(tile[2*threadIdx.y][2*threadIdx.x] > col[2*threadIdx.y][k]+row[k][2*threadIdx.x])
      tile[2*threadIdx.y][2*threadIdx.x]= col[2*threadIdx.y][k]+row[k][2*threadIdx.x];
    if(tile[2*threadIdx.y][2*threadIdx.x+1] > col[2*threadIdx.y][k]+row[k][2*threadIdx.x+1])
      tile[2*threadIdx.y][2*threadIdx.x+1]= col[2*threadIdx.y][k]+row[k][2*threadIdx.x+1];
    if(tile[2*threadIdx.y+1][2*threadIdx.x] > col[2*threadIdx.y+1][k]+row[k][2*threadIdx.x])
      tile[2*threadIdx.y+1][2*threadIdx.x]= col[2*threadIdx.y+1][k]+row[k][2*threadIdx.x];
    if(tile[2*threadIdx.y+1][2*threadIdx.x+1] > col[2*threadIdx.y+1][k]+row[k][2*threadIdx.x+1])
      tile[2*threadIdx.y+1][2*threadIdx.x+1]= col[2*threadIdx.y+1][k]+row[k][2*threadIdx.x+1];
    __syncthreads();
  }

  g[y_t*N + x_t]= tile[2*threadIdx.y][2*threadIdx.x];
  g[y_t*N + x_t1]= tile[2*threadIdx.y][2*threadIdx.x+1];
  g[y_t1*N + x_t]= tile[2*threadIdx.y+1][2*threadIdx.x];
  g[y_t1*N + x_t1]= tile[2*threadIdx.y+1][2*threadIdx.x+1];
}


template<int n>
__global__ void phase1m2(int* g, const int pstart, const int N){
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

template<int n>
__global__ void phase2m2(int* g, const int pstart, const int primary_n, const int N){
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

template<int n>
__global__ void phase3m2(int* g, const int pstart, const int primary_n, const int N){
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
