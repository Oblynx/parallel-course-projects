//! @file Parallel bitonic sort implemented with C++11 std threads
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include "thread_pool.h"
using namespace std;

#define UP 1
#define DOWN 0

class RandArray{
  public:
  //! Call workers to initialize random array
  RandArray(int threadN, int numN): threadN_(threadN), numN_(numN), threadRange_(numN/threadN),
      data_(new int[numN]){
    srand(1);
    vector<thread> threads;
    threads.reserve(threadN);
    for(int i=0; i<threadN; i++) threads.emplace_back(&RandArray::construct,this,i);
    for(int i=0; i<threadN; i++) threads[i].join();
  }
  void sort();
  //! Check result correctness. Could also be a simple out-of-order search of course
  int check();
  void print(){
    for(int i=0; i<numN_; i++) cout << data_[i] << ' ';
    cout << '\n';
  }

  private:
  //! Thread callback for creating random array slice
  void construct(int frame);
  void recBitonicSort(int lo, int cnt, int direct);
  void bitonicMerge(int lo, int cnt, int direct);
  inline void exchange(int a, int b) {
    int tmp;
    tmp=data_[a], data_[a]=data_[b], data_[b]=tmp;
  }
  int threadN_, numN_;
  //! Size of array slice for each thread
  int threadRange_;
  //! Data + copy for independent sorting to compare times
  unique_ptr<int[]> data_;
  const int seqThres_= 1;
};

int main(int argc, char** argv){
  if (argc<3){
    cout << "Parallel bitonic sort using STD threads.\nUsage:\t" << argv[0]
         << " <log2 num of elements> <log2 num of threads>\n\n";
    return 1;
  }
  const int logThreadN= strtol(argv[2], NULL, 10);
  const int logNumN= strtol(argv[1], NULL, 10);
  if (logThreadN > 8){
    cout << "Max thread number: 2^8\n";
    return 2;
  }else if (logNumN > 24){
    cout << "Max elements number: 2^24\n";
    return 3;
  }
  int threadN, numN;
  numN= 1<<logNumN;
  threadN= (logThreadN>logNumN)? 1<<logNumN:1<<logThreadN; // Not more threads than elements

  // Input done, let's get the threads running...
  ThreadPool workers(threadN);
  RandArray array(threadN, numN);
  array.print();
  array.sort();
  array.print();
  return array.check();
}

void RandArray::construct(int frame){
  const int start= frame*threadRange_, end= (frame+1)*threadRange_;
  // Hopefully the C++ stdlib implementation of rand() has no data races, unlike the C version
  // As mentioned here: http://www.cplusplus.com/reference/cstdlib/rand/
  for(int i=start; i<end; i++) data_[i]= rand() %20;
}

//!  imperative version of bitonic sort
void RandArray::sort(){
  int i,j,k;
  for (k=2; k<=numN_; k=k<<1) {
    for (j=k>>1; j>0; j=j>>1) {
      for (i=0; i<numN_; i++) {
        int ij=i^j;
        if ((ij)>i) {
          if ((i&k)==0 && data_[i] > data_[ij]) 
            exchange(i,ij);
          if ((i&k)!=0 && data_[i] < data_[ij])
            exchange(i,ij);
        }
      }
    }
  }
} 

//int compare (const void * a, const void * b) {return ( *(int*)a - *(int*)b );}
int RandArray::check(){
//  qsort(checkCpy_.get(), numN_, sizeof(int), compare);
  for(int i=0; i<numN_-1; i++) if(data_[i] > data_[i+1]) return false;
  return true;
}







