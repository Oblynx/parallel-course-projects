//! @file Parallel bitonic sort implemented with C++11 std threads
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include "thread_pool.h"
#ifdef __DEBUG__
#define COUT cout
#define DBG_PRINTF printf //if(!(ID & ~3u)) printf
#else
#define COUT while(0) cout
#define DBG_PRINTF while(0) printf
#endif
using namespace std;

int compUP (const void *a, const void *b) {return ( *(unsigned*)a - *(unsigned*)b );}
int compDN (const void *a, const void *b) {return ( *(unsigned*)b - *(unsigned*)a );}

class RandArray{
public:
  //! Call workers to initialize random array
  RandArray(unsigned numN, ThreadPool& workers): numN_(numN), seqThres_(numN/workers.workers()>>1),
      exchangeBuffLength_(2*numN_/seqThres_), workers_(workers), data_(new unsigned[numN]),
      nodeStatus_(new unsigned char[2*numN]),
      exchangeComplete_(new unsigned char[exchangeBuffLength_]), serial_(0){
#ifndef BATCH_EXPERIMENTS
    cout << "Constructing RandArray\n";
#endif
    srand(1);
    const unsigned smallProblemThres= (seqThres_ > numN_)? seqThres_: numN_;
    for(unsigned i=0; i< exchangeBuffLength_; i++) exchangeComplete_[i]= 0;
    vector<future<void>> results;
    results.reserve(numN_/smallProblemThres);
    for(unsigned i=0; i< numN_/smallProblemThres; i++)
      results.push_back(workers_.schedule(&RandArray::construct, this,i,smallProblemThres));
    for(unsigned i=0; i< numN_/smallProblemThres; i++) results[i].get();
  }
  void sort();
  void seqSort(){
    qsort(data_.get(), numN_, sizeof(unsigned), compUP);
  }
  //! Check result correctness. Could also be a simple out-of-order search of course
  int check();
  void print(){
    for(unsigned i=0; i<numN_; i++) cout << data_[i] << ' ';
    cout << '\n';
  }
  ~RandArray(){
#ifndef BATCH_EXPERIMENTS
    cout << "Destroying RandArray\n";
#endif
  }
private:
  //! Thread callback for creating random array slice
  void construct(const unsigned frame, const unsigned taskRange);
  void recBitonicSort(const unsigned lo, const unsigned cnt, const int direct, const unsigned ID);
  void sortFinalize(const unsigned cnt, const unsigned ID);
  void bitonicMerge(const unsigned lo, const unsigned cnt, const int direct, const unsigned ID);
  void sortContin(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID);
  void mergeContin(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID,
      const unsigned prereqStart, const unsigned prereqEnd);
  void mergeFinalize(const unsigned cnt, const unsigned ID);
  inline void exchange(const unsigned a, const unsigned b) {
    unsigned tmp;
    tmp=data_[a], data_[a]=data_[b], data_[b]=tmp;
  }

  const unsigned numN_, seqThres_, exchangeBuffLength_;
  //! Size of array slice for each thread
  ThreadPool& workers_;
  unique_ptr<unsigned[]> data_;
  unique_ptr<unsigned char[]> nodeStatus_;
  unique_ptr<unsigned char[]> exchangeComplete_;
  atomic<unsigned> serial_;
  static const unsigned ASCENDING, DESCENDING;
  //! Signal that all tasks have finished
  std::mutex finishMut_;
  std::condition_variable finishCnd_;
  bool finished_;
};
//const unsigned RandArray::seqThres_= 1<<19;
const unsigned RandArray::ASCENDING=1, RandArray::DESCENDING=0;

int main(int argc, char** argv){
  if (argc<3){
    cout << "Parallel bitonic sort using STD threads.\nUsage:\t" << argv[0]
         << " <log2 num of elements> <log2 num of threads>\n\n";
    return 1;
  }
  const unsigned logThreadN= strtol(argv[2], NULL, 10);
  const unsigned logNumN= strtol(argv[1], NULL, 10);
  if (logThreadN > 8){
    cout << "Max thread number: 2^8\n";
    return 2;
  }else if (logNumN > 27){
    cout << "Max elements number: 2^24\n";
    return 3;
  }
  unsigned threadN, numN;
  numN= 1<<logNumN;
  threadN= (logThreadN>logNumN)? 1<<logNumN:1<<logThreadN; // Not more threads than elements
  // Input done, let's get the threads running...

  ThreadPool workers(threadN);
  auto start= chrono::system_clock::now();
  RandArray array(numN, workers);
  chrono::duration<double> duration= chrono::system_clock::now()-start;
#ifndef BATCH_EXPERIMENTS
  cout<<"--> Array constructed in "<<duration.count()*1000<<"ms\n";                                 
#else
  cout<<duration.count()*1000<<' ';
#endif
  //array.print();
  start= chrono::system_clock::now();  
  if(argc == 4){
    if(strcmp(argv[3],"-qsort") || strcmp(argv[3], "-seq")) array.seqSort();
    else array.sort();
  }
  else array.sort();
  duration= chrono::system_clock::now()-start;
#ifndef BATCH_EXPERIMENTS
  cout<<"--> Array sorted in "<<duration.count()*1000<<"ms\n";
#else
  cout<<duration.count()*1000<<'\n';
#endif
  //array.print();
  return array.check();
}

void RandArray::construct(const unsigned frame, const unsigned taskRange){
  const unsigned start= frame*taskRange, end= (frame+1)*taskRange;
  // Hopefully the C++ stdlib implementation of rand() has no data races, unlike the C version
  // As mentioned here: http://www.cplusplus.com/reference/cstdlib/rand/
  for(unsigned i=start; i<end; i++) data_[i]= rand() %20;
}
void RandArray::sort(){
  finished_= false;
  COUT<<"Scheduling tasks...\n";
  recBitonicSort(0,numN_,ASCENDING,0);
  COUT << "All tasks scheduled!\n";
  std::unique_lock<std::mutex> lk(finishMut_);
  finishCnd_.wait(lk, [=] { return finished_; });
  //workers_.waitFinish();
  COUT << "Waited as well\n";
}
int RandArray::check(){
  for(unsigned i=0; i<numN_-1; i++) if(data_[i] > data_[i+1]){
#ifndef BATCH_EXPERIMENTS
    std::cout <<"\n\t   ####################   \
                 \n\t--| ### Test FAIL! ### |--\
                 \n\t   ####################   \n\n";
#endif
    return false;
  }
#ifndef BATCH_EXPERIMENTS
  std::cout <<"\n\t   ####################   \
               \n\t--| ### Test PASS! ### |--\
               \n\t   ####################   \n\n";
#endif
  return true;
}

// Only insert prereq if this is a left-node (worker==true)
void RandArray::recBitonicSort(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID){
  nodeStatus_[ID]= 0;
  if (cnt>seqThres_) {
    DBG_PRINTF("[recBitonicSort]: recursing\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    unsigned k=cnt/2;
    workers_.schedule(&RandArray::recBitonicSort, this,lo, k, ASCENDING, ID+1);
    recBitonicSort(lo+k, k, DESCENDING, ID+cnt);
    workers_.schedule([=] (){
        workers_.schedule(&RandArray::sortContin,this,lo,cnt,dir,ID);
    });
  } else{
    DBG_PRINTF("[recBitonicSort]: LEAF\t\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    if(dir) qsort(data_.get()+lo, cnt, sizeof(unsigned),compUP);
    else qsort(data_.get()+lo, cnt, sizeof(unsigned),compDN);
    nodeStatus_[ID]|= 1;
  }
}
// Depends on ID+1, ID+cnt having been sorted
void RandArray::sortContin(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID){
  if((nodeStatus_[ID+1] & 1) && (nodeStatus_[ID+cnt] & 1)){
    DBG_PRINTF("[sortContin]: continuing\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    bitonicMerge(lo,cnt,dir, ID);
    workers_.schedule([=]{workers_.schedule([=]{
          workers_.schedule(&RandArray::sortFinalize, this, cnt, ID);
    });});
    return;
  }
  //If dependency isn't complete, reschedule
  workers_.schedule([=] {workers_.schedule(&RandArray::sortContin, this,lo,cnt,dir,ID);});
}
//! Signal this task is complete and erase its dependencies, which are no longer needed 
void RandArray::sortFinalize(const unsigned cnt, const unsigned ID){
  if(nodeStatus_[ID] & 4){
    DBG_PRINTF("[sortFinalize]: Making 2\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    nodeStatus_[ID]|= 1;
    // If this is the root node, signal algorithm completion
    if(ID == 0){{
        std::lock_guard<std::mutex> lk(finishMut_);
        finished_= true;
      }
      finishCnd_.notify_all();
    }
    return;
  }
  //reschedule
  workers_.schedule([=]{workers_.schedule(&RandArray::sortFinalize, this, cnt, ID);});
}

//! For small problems, synchronously merge; for larger sizes, launch asynchronous merging tasks
void RandArray::bitonicMerge(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID){
  if (cnt>1) {
    nodeStatus_[ID]&= ~4u; //clear merge flag
    unsigned k= cnt>>1;
    const unsigned smallProblemThres= (seqThres_ < k)? seqThres_: k;
    if (smallProblemThres < k){
      const unsigned chunkNumber= k/smallProblemThres;
      // Request a range of serial numbers
      const unsigned serialStart= serial_.fetch_add(chunkNumber);
      const unsigned serialEnd= serialStart+chunkNumber;
      for(unsigned i=0; i< chunkNumber; i++){
        workers_.schedule([=] (const unsigned serial){
          if(exchangeComplete_[serial]){
            cout << "$$$ FATAL ERROR: Chunk queue length exceeded\n";
            exit(1);
          }
          const unsigned start= lo+i*smallProblemThres, end= lo+(i+1)*smallProblemThres;
          for(unsigned i=start; i<end; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k);
          exchangeComplete_[serial]= 1;
        }, (serialStart+i)%exchangeBuffLength_);
      }
      // Schedule the rest of bitonicMerge
      workers_.schedule(&RandArray::mergeContin, this,lo,cnt,dir,ID, serialStart,serialEnd);
    } else {  // If problem is too small, run everything sequentially
      DBG_PRINTF("[bitonicMerge]: Making 1\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
      for (unsigned i=lo; i<lo+k; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k);
      bitonicMerge(lo, k, dir, ID+1);
      bitonicMerge(lo+k, k, dir, ID+cnt);
      // Signal merge completion
      nodeStatus_[ID]|= 4;
    }
  }
}

void RandArray::mergeContin(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID,
    const unsigned prereqStart, const unsigned prereqEnd){
  for(unsigned serial= prereqStart; serial< prereqEnd; serial++){
    if(!exchangeComplete_[serial%exchangeBuffLength_]){
      // Schedule-to-reschedule?
      //workers_.schedule([=]{
          workers_.schedule(&RandArray::mergeContin, this,lo,cnt,dir, ID, prereqStart, prereqEnd);
      //});
      return;
    }
  }
  for(unsigned serial= prereqStart; serial< prereqEnd; serial++)
    exchangeComplete_[serial%exchangeBuffLength_]= 0;
  // All prerequisites have completed!
  DBG_PRINTF("[mergeContin]: Schedul_merges\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  const unsigned k= cnt>>1;
  workers_.schedule(&RandArray::bitonicMerge, this, lo,k,dir, ID+1);
  bitonicMerge(lo+k, k, dir, ID+cnt);
  // Schedule a rescheduling of mergeFinalize (which signals merge completion for this ID)
  workers_.schedule([=] (){
    workers_.schedule(&RandArray::mergeFinalize, this, cnt, ID);
  });
}
// After previous bitonic merges have completed, signal completion
void RandArray::mergeFinalize(const unsigned cnt, const unsigned ID){
  if((nodeStatus_[ID+1] & 4) && (nodeStatus_[ID+cnt] & 4)){
    DBG_PRINTF("[mergeFinalize]: Making 1\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    nodeStatus_[ID]|= 4;
    return;
  }
  //reschedule
  workers_.schedule([=] {workers_.schedule(&RandArray::mergeFinalize, this, cnt, ID);});
}





