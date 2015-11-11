//! @file Parallel bitonic sort implemented with C++11 std threads
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include "thread_pool.h"
#include <tbb/concurrent_hash_map.h>
using namespace std;
typedef tbb::concurrent_hash_map<int,char> ConcurMap;
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
  void recBitonicSort(int lo, int cnt, int direct,int ID);
  void bitonicMerge(int lo, int cnt, int direct);
  void continuation(int lo, int cnt, int dir, int ID);
  inline void exchange(int a, int b) {
    int tmp;
    tmp=data_[a], data_[a]=data_[b], data_[b]=tmp;
  }
  int threadN_, numN_;
  //! Size of array slice for each thread
  int threadRange_;
  ThreadPool& workers_;
  unique_ptr<int[]> data_;
  ConcurMap taskComplete_;
  const int seqThres_= 1<<0, ASCENDING=1, DESCENDING=0;
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
  array.print();
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

// Each continuation always depends on ID+1, ID+cnt
void RandArray::continuation(int lo, int cnt, int dir, int ID){
  printf("[continuation]: Enter\t\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  bool notReady[2]={false,false};
  {
    ConcurMap::const_accessor ac;
    taskComplete_.find(ac, ID+1);
    notReady[0]= ac->second == 0;
    ac.release();
    taskComplete_.find(ac, ID+cnt);
    notReady[1]= ac->second == 0;
  }//If dependency isn't complete, reschedule
  if(notReady[0] || notReady[1]){
    printf("[continuation]: rescheduling\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    workers_.schedule(&RandArray::continuation, this,lo,cnt,dir,ID);
    return;
  }
  printf("[continuation]: continuing\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  bitonicMerge(lo,cnt,dir);
  {  // Signal this task is complete and erase its dependencies, which are no longer useful
    ConcurMap::accessor ac;
    taskComplete_.find(ac, ID);
    ac->second= 1;
  }
  taskComplete_.erase(ID+1);
  taskComplete_.erase(ID+cnt);
}

// Only insert prereq if this is a left-node (worker==true)
void RandArray::recBitonicSort(int lo, int cnt, int dir, int ID){
  taskComplete_.insert(make_pair(ID,0));
  if (cnt>seqThres_) {
    printf("[recBitonicSort]: recursing\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    int k=cnt/2;
    workers_.schedule(&RandArray::recBitonicSort, this,lo, k, ASCENDING, ID+1);
    recBitonicSort(lo+k, k, DESCENDING, ID+cnt);
    workers_.schedule(&RandArray::continuation,this,lo,cnt,dir,ID);
  } else{
    printf("[recBitonicSort]: LEAF\t\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    if(dir) qsort(data_.get()+lo, cnt, sizeof(int),compUP);
    else qsort(data_.get()+lo, cnt, sizeof(int),compDN);
    {
      ConcurMap::accessor ac;
      taskComplete_.find(ac,ID);
      ac->second= 2;
    }
  }
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

void RandArray::sort(){
  cout<<"Scheduling tasks...\n";
  recBitonicSort(0,numN_,ASCENDING,0);
  cout << "All tasks scheduled!\n";
  workers_.waitFinish();
  cout << "Waited as well\n";
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







