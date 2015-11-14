//! @file Parallel bitonic sort implemented with C++11 std threads
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include "tbb-4.4/concurrent_hash_map.h"
#include "thread_pool.h"
#ifdef __DEBUG__
#define COUT cout
#define DBG_PRINTF if(!(ID & ~3u)) printf
#else
#define COUT while(0) cout
#define DBG_PRINTF while(0) printf
#endif
using namespace std;

typedef tbb::concurrent_hash_map<unsigned,unsigned char> ConcurMap;

class RandArray{
public:
  //! Call workers to initialize random array
  RandArray(unsigned threadN, unsigned numN, ThreadPool& workers): threadN_(threadN), numN_(numN),
      workers_(workers), data_(new unsigned[numN]), taskComplete_(round(0.1*numN_)),
      exchangeComplete_(round(0.6*numN_/(seqThres_<<2))), nodeStatus_(numN_>>3), serial_(0){
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

  const unsigned threadN_, numN_;
  //! Size of array slice for each thread
  ThreadPool& workers_;
  unique_ptr<unsigned[]> data_;
  ConcurMap taskComplete_, exchangeComplete_, nodeStatus_;
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
    std::cout <<"\t--| ### TEST FAILS! ### |--\n";
    return false;
  }
  std::cout <<"\t--| ### TEST PASSES! ### |--\n";
  return true;
}

// Only insert prereq if this is a left-node (worker==true)
void RandArray::recBitonicSort(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID){
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
    // TODO: Probably unnecessary check
    if(!taskComplete_.insert(make_pair(ID,2))){
      ConcurMap::accessor acMod;
      if(taskComplete_.find(acMod, ID)) acMod->second= 2;
      else throw new domain_error("[sort]: ERROR! Current node #"+
                                  to_string(ID)+" was JUST deleted by someone else!\n");
    }
  }
}
// Depends on ID+1, ID+cnt having been sorted
void RandArray::sortContin(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID){
  { //OnlyOnce
    ConcurMap::const_accessor ac;
    if(nodeStatus_.find(ac,ID)) if(ac->second & 1) return; 
  }
  ConcurMap::const_accessor ac;
  if(taskComplete_.find(ac, ID+1))
    if(ac->second == 2){
      ac.release();
      if(taskComplete_.find(ac, ID+cnt))
        if(ac->second == 2){
          ac.release();
          { //OnlyOnce
            if(!nodeStatus_.insert(make_pair(ID,1))) throw new domain_error("[sortContin]: OnlyOnce ERROR\n");
          }
          DBG_PRINTF("[sortContin]: continuing\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
          if (ID==0) printf("\n\n\n");
          bitonicMerge(lo,cnt,dir, ID);
          workers_.schedule(&RandArray::sortFinalize, this, cnt, ID);
          return;
        }
    }
  //If dependency isn't complete, reschedule
  //DBG_PRINTF("[sortContin]: rescheduling\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  workers_.schedule(&RandArray::sortContin, this,lo,cnt,dir,ID);
}
//! Signal this task is complete and erase its dependencies, which are no longer needed 
void RandArray::sortFinalize(const unsigned cnt, const unsigned ID){
  { //OnlyOnce
    ConcurMap::const_accessor ac;
    if(nodeStatus_.find(ac,ID)) if(ac->second & 2) return;
  }
  ConcurMap::const_accessor ac;
  if(taskComplete_.find(ac, ID))
    if(ac->second == 1){
      ac.release();
      { //OnlyOnce
        ConcurMap::accessor ac;
        if(nodeStatus_.find(ac, ID)) ac->second|= 2;
        else throw new domain_error("[sortFinalize]: OnlyOnce ERROR\n");
      }
      DBG_PRINTF("[sortFinalize]: Making 2\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
      ConcurMap::accessor acMod;
      if(taskComplete_.find(acMod, ID)) acMod->second= 2;
      else throw new domain_error("[sort]: ERROR! Current node #"+
                                  to_string(ID)+" doesn't exist yet!\n");
      acMod.release();
      taskComplete_.erase(ID+1);
      taskComplete_.erase(ID+cnt);
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
  //DBG_PRINTF("[sortFinalize]: rescheduling\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  workers_.schedule(&RandArray::sortFinalize, this, cnt, ID);
}

//! For small problems, synchronously merge; for larger sizes, launch asynchronous merging tasks
void RandArray::bitonicMerge(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID){
  if (cnt>1) {
    { //OnlyOnce
      ConcurMap::accessor ac;
      if(nodeStatus_.find(ac, ID)) ac->second&= 3; //Clear merge flags
    }
    unsigned k= cnt>>1;
    const unsigned smallProblemThres= (seqThres_<<0 < k)? seqThres_<<0: k;
    if (smallProblemThres < k){
      const unsigned chunkNumber= k/smallProblemThres;
      // Request a range of serial numbers
      const unsigned serialStart= serial_.fetch_add(chunkNumber);
      const unsigned serialEnd= serialStart+chunkNumber;
      for(unsigned i=0; i< chunkNumber; i++){
        workers_.schedule([=] (const unsigned serial){
          const unsigned start= lo+i*smallProblemThres, end= lo+(i+1)*smallProblemThres;
          for(unsigned i=start; i<end; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k);
          this->exchangeComplete_.insert(make_pair(serial,1));
        }, serialStart+i);
      }
      // Schedule the rest of bitonicMerge
      workers_.schedule(&RandArray::mergeContin, this,lo,cnt,dir,ID, serialStart,serialEnd);
    } else {  // If problem is too small, run everything sequentially
      DBG_PRINTF("[bitonicMerge]: Making 1\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
      for (unsigned i=lo; i<lo+k; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k);
      bitonicMerge(lo, k, dir, ID+1);
      bitonicMerge(lo+k, k, dir, ID+cnt);
      // Signal merge completion: insert ID or modify, if it already exists
      if(!taskComplete_.insert(make_pair(ID,1))){
        ConcurMap::accessor ac;
        if(taskComplete_.find(ac, ID)) ac->second= 1;
        else throw new domain_error("[merge]: ERROR! Current node #"+
                                    to_string(ID)+" was JUST deleted by someone else!\n");
      }
    }
  }
}

void RandArray::mergeContin(const unsigned lo, const unsigned cnt, const int dir, const unsigned ID,
    const unsigned prereqStart, const unsigned prereqEnd){
  { //OnlyOnce
    ConcurMap::const_accessor ac;
    if(nodeStatus_.find(ac, ID)) if(ac->second & 4) return;
  }
  ConcurMap::const_accessor ac;
  for(unsigned serial= prereqStart; serial< prereqEnd; serial++){
    if(!exchangeComplete_.find(ac, serial)){
      // TODO: Schedule-to-reschedule? (perf)
      workers_.schedule(&RandArray::mergeContin, this,lo,cnt,dir, ID, prereqStart, prereqEnd);
      return;
    }
    ac.release();
  }
  { //OnlyOnce
    ConcurMap::accessor ac;
    if(nodeStatus_.find(ac,ID)) ac->second|= 4;
    else throw new domain_error("[mergeContin]: OnlyOnce ERROR\n");
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
  { //OnlyOnce
    ConcurMap::const_accessor ac;
    if(nodeStatus_.find(ac, ID)) if(ac->second & 8) return;
  }
  ConcurMap::const_accessor ac;
  if(taskComplete_.find(ac, ID+1))
    if(ac->second == 1){
      ac.release();
      if(taskComplete_.find(ac, ID+cnt))
        if(ac->second == 1){
          ac.release();
          { //OnlyOnce
            ConcurMap::accessor ac;
            if(nodeStatus_.find(ac,ID)) ac->second|= 8;
          }
          DBG_PRINTF("[mergeFinalize]: Making 1\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
          if(!taskComplete_.insert(make_pair(ID,1))){
            ConcurMap::accessor acMod;
            if(taskComplete_.find(acMod, ID)) acMod->second= 1;
            else throw new domain_error("[mergeFin]: ERROR! Current node #"+
                                        to_string(ID)+" was JUST deleted by someone else!\n");
          }
          return;
        }
    }
  ac.release();
  //reschedule
  //DBG_PRINTF("[mergeFinalize]: Reschedule\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  workers_.schedule(&RandArray::mergeFinalize, this, cnt, ID);
}





