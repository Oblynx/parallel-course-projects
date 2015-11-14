//! @file Parallel bitonic sort implemented with C++11 std threads
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include "tbb-4.4/concurrent_unordered_map.h"
#include "thread_pool.h"
#ifdef __DEBUG__
#define COUT cout
#define DBG_PRINTF printf //if(!(ID & ~3u)) printf
#else
#define COUT while(0) cout
#define DBG_PRINTF while(0) printf
#endif
using namespace std;

typedef tbb::concurrent_unordered_map<unsigned,unsigned char> ConcurMap;

class ConstIter: public iterator<input_iterator_tag, pair<unsigned,unsigned char>>{
  typedef pair<unsigned, unsigned char> T;
  T c;
  unsigned& ID;
public:
  ConstIter(unsigned ID, unsigned char s): c(make_pair(ID,s)), ID(c.first) {}
  ConstIter(const ConstIter& ot)= default;
  ConstIter& operator++() {ID++; return *this;}
  ConstIter operator++(int) {ConstIter tmp(*this); operator++(); return tmp;}
  bool operator==(const ConstIter& rhs) {return ID==rhs.ID;}
  bool operator!=(const ConstIter& rhs) {return ID!=rhs.ID;}
  T& operator*() {return c;}
};

class RandArray{
public:
  //! Call workers to initialize random array
  RandArray(unsigned numN, ThreadPool& workers): numN_(numN),
      workers_(workers), data_(new unsigned[numN]),
      nodeStatus_(ConstIter(0,0), ConstIter(2*numN_-1,0), 1*numN_),
      exchangeComplete_(numN_/(seqThres_<<2)), serial_(0){
    cout << "Constructing RandArray\n";
    srand(1);
    const unsigned smallProblemThres= (seqThres_<<2 > numN_)? seqThres_<<2: numN_;
    vector<future<void>> results;
    results.reserve(numN_/smallProblemThres);
    for(unsigned i=0; i< numN_/smallProblemThres; i++)
      results.push_back(workers_.schedule(&RandArray::construct, this,i,smallProblemThres));
    for(unsigned i=0; i< numN_/smallProblemThres; i++) results[i].get();
  }
  void sort();
  //! Check result correctness. Could also be a simple out-of-order search of course
  int check();
  void print(){
    for(unsigned i=0; i<numN_; i++) cout << data_[i] << ' ';
    cout << '\n';
  }
  ~RandArray(){ cout << "Destroying RandArray\n"; }
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

  const unsigned numN_;
  //! Size of array slice for each thread
  ThreadPool& workers_;
  unique_ptr<unsigned[]> data_;
  ConcurMap nodeStatus_, exchangeComplete_;
  atomic<unsigned> serial_;
  static const unsigned seqThres_, ASCENDING, DESCENDING;
  //! Signal that all tasks have finished
  std::mutex finishMut_;
  std::condition_variable finishCnd_;
  bool finished_;
};
const unsigned RandArray::seqThres_= 1<<0, RandArray::ASCENDING=1, RandArray::DESCENDING=0;

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
  }else if (logNumN > 24){
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
  cout<<"--> Array constructed in "<<duration.count()*1000<<"ms\n";
  //array.print();
  start= chrono::system_clock::now();
  array.sort();
  duration= chrono::system_clock::now()-start;
  cout<<"--> Array sorted in "<<duration.count()*1000<<"ms\n";
  //array.print();
  return array.check();
}

int compUP (const void *a, const void *b) {return ( *(unsigned*)a - *(unsigned*)b );}
int compDN (const void *a, const void *b) {return ( *(unsigned*)b - *(unsigned*)a );}

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
  workers_.waitFinish();
  COUT << "Waited as well\n";
}
int RandArray::check(){
  //  qsort(checkCpy_.get(), numN_, sizeof(unsigned), compare);
  for(unsigned i=0; i<numN_-1; i++) if(data_[i] > data_[i+1]){
    std::cout <<"\n\t   ####################   \
                 \n\t--| ### Test FAIL! ### |--\
                 \n\t   ####################   \n\n";
    return false;
  }
  std::cout <<"\n\t   ####################   \
               \n\t--| ### Test PASS! ### |--\
               \n\t   ####################   \n\n";
  return true;
}

// Only insert prereq if this is a left-node (worker==true)
void RandArray::recBitonicSort(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID){
  //nodeStatus_[ID]= 0;
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
    workers_.schedule(&RandArray::sortFinalize, this, cnt, ID);
    return;
  }
  //If dependency isn't complete, reschedule
  workers_.schedule(&RandArray::sortContin, this,lo,cnt,dir,ID);
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
  workers_.schedule(&RandArray::sortFinalize, this, cnt, ID);
}

//! For small problems, synchronously merge; for larger sizes, launch asynchronous merging tasks
void RandArray::bitonicMerge(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID){
  if (cnt>1) {
    nodeStatus_[ID]&= ~4u; //clear merge flag
    unsigned k= cnt>>1;
    const unsigned smallProblemThres= (seqThres_<<2 < k)? seqThres_<<2: k;
    if (smallProblemThres < k){
      const unsigned chunkNumber= k/smallProblemThres;
      // Request a range of serial numbers
      const unsigned serialStart= serial_.fetch_add(chunkNumber);
      const unsigned serialEnd= serialStart+chunkNumber;
      for(unsigned i=0; i< chunkNumber; i++){
        exchangeComplete_[serialStart+i]= 0;
        workers_.schedule([=] (const unsigned serial){
          const unsigned start= lo+i*smallProblemThres, end= lo+(i+1)*smallProblemThres;
          for(unsigned i=start; i<end; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k);
          this->exchangeComplete_[serial]= 1;
        }, serialStart+i);
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
    if(!(exchangeComplete_[serial] & 1)){
      // TODO: Schedule-to-reschedule?
      workers_.schedule(&RandArray::mergeContin, this,lo,cnt,dir, ID, prereqStart, prereqEnd);
      return;
    }
  }
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
  workers_.schedule(&RandArray::mergeFinalize, this, cnt, ID);
}





