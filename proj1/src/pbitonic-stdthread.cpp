//! @file Parallel bitonic sort implemented with C++11 std threads
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include "thread_pool.h"
using namespace std;

#define UP 1
#define DOWN 0

class RandArray{
  public:
  //! Call workers to initialize random array
  RandArray(int threadN, int numN, ThreadPool& workers): threadN_(threadN), numN_(numN),
      threadRange_(numN/threadN), workers_(workers), data_(new int[numN]){
    cout << "Constructing RandArray\n";
    srand(1);
    vector<future<void>> results;
    results.reserve(threadN);
    for(int i=0; i<threadN; i++) results[i]= workers_.schedule(&RandArray::construct,this,i);
    for(int i=0; i<threadN; i++) results[i].get();
  }
  void sort();
  //! Check result correctness. Could also be a simple out-of-order search of course
  int check();
  void print(){
    for(int i=0; i<numN_; i++) cout << data_[i] << ' ';
    cout << '\n';
  }

  ~RandArray(){ cout << "Destroying RandArray\n"; }
  private:
  //! Thread callback for creating random array slice
  void construct(int frame);
  void recBitonicSort(int lo, int cnt, int direct);
  void bitonicMerge(int lo, int cnt, int direct);
  void loopsort();
  inline void exchange(int a, int b) {
    int tmp;
    tmp=data_[a], data_[a]=data_[b], data_[b]=tmp;
  }
  int threadN_, numN_;
  //! Size of array slice for each thread
  int threadRange_;
  ThreadPool& workers_;
  unique_ptr<int[]> data_;
  const int seqThres_= 1<<10, ASCENDING=1, DESCENDING=0;
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
  auto start= chrono::system_clock::now();
  RandArray array(threadN, numN, workers);
  chrono::duration<double> duration= chrono::system_clock::now()-start;
  cout<<"--> Array constructed in "<<duration.count()*1000<<"ms\n";
  //array.print();
  start= chrono::system_clock::now();
  array.sort();
  duration= chrono::system_clock::now()-start;
  cout<<"--> Array sorted in "<<duration.count()*1000<<"ms\n";
  //array.print();
  return array.check();
}

int compUP (const void * a, const void * b) {return ( *(int*)a - *(int*)b );}
int compDN (const void * a, const void * b) {return ( *(int*)b - *(int*)a );}

void RandArray::construct(int frame){
  const int start= frame*threadRange_, end= (frame+1)*threadRange_;
  // Hopefully the C++ stdlib implementation of rand() has no data races, unlike the C version
  // As mentioned here: http://www.cplusplus.com/reference/cstdlib/rand/
  for(int i=start; i<end; i++) data_[i]= rand() %20;
}

//!  imperative version of bitonic sort
void RandArray::loopsort(){
  int k,j,frame;
  vector<future<void>> results;
  results.reserve(log2(numN_)*log2(numN_)*threadN_);
  for (k=2; k<=numN_; k=k<<1) {
    for (j=k>>1; j>0; j=j>>1) {
      for (frame=0; frame<threadN_; frame++) {
        const int start= frame*threadRange_, end= (frame+1)*threadRange_;
        results.emplace_back(workers_.schedule([=] (){
          for(int i=start; i<end; i++){
            int ij=i^j;
            if ((ij)>i) {
              if ((i&k)==0 && data_[i] > data_[ij]) 
                exchange(i,ij);
              if ((i&k)!=0 && data_[i] < data_[ij])
                exchange(i,ij);
            }
          }
        }));
      }
    }
  }
  for(auto&& result: results)result.get();
}

void RandArray::sort(){
  recBitonicSort(0,numN_,ASCENDING);
}

void RandArray::recBitonicSort(int lo, int cnt, int dir) {
  if (cnt>seqThres_) {
    int k=cnt/2;
    future<void> sortLow= workers_.schedule(&RandArray::recBitonicSort, this, lo, k, ASCENDING);
    //recBitonicSort(lo, k, ASCENDING);
    recBitonicSort(lo+k, k, DESCENDING);
    //Deadlock! Need to signal on condition instead?
    workers_.schedule([=] (future<void>&& task1){
      task1.get();
      bitonicMerge(lo,cnt,dir);
    },move(sortLow));
    //task.get();
    //bitonicMerge(lo, cnt, dir);
  } else if(dir) qsort(data_.get()+lo, cnt, sizeof(int),compUP);
  else qsort(data_.get()+lo, cnt, sizeof(int),compDN);

}
void RandArray::bitonicMerge(int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    int i;
    for (i=lo; i<lo+k; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k);
    bitonicMerge(lo, k, dir);
    bitonicMerge(lo+k, k, dir);
  }
}

int RandArray::check(){
//  qsort(checkCpy_.get(), numN_, sizeof(int), compare);
  for(int i=0; i<numN_-1; i++) if(data_[i] > data_[i+1]){
    std::cout <<"TEST FAILS!\n";
    return false;
  }
  std::cout <<"TEST PASSES!\n";
  return true;
}







