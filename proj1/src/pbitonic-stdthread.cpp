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
#define DBG_PRINTF printf
#else
#define COUT while(0) cout
#define DBG_PRINTF while(0) printf
#endif
using namespace std;

typedef tbb::concurrent_hash_map<int,char> ConcurMap;

class RandArray{
public:
  //! Call workers to initialize random array
  RandArray(int threadN, int numN, ThreadPool& workers): threadN_(threadN), numN_(numN),
      workers_(workers), data_(new int[numN]), taskComplete_(round(0.1*numN_)),
      exchangeComplete_(round(0.6*numN_/(seqThres_<<2))), serial_(0){
    cout << "Constructing RandArray\n";
    srand(1);
    const int smallProblemThres= (seqThres_<<2 > numN_)? seqThres_<<2: numN_;
    vector<future<void>> results;
    results.reserve(numN_/smallProblemThres);
    for(int i=0; i< numN_/smallProblemThres; i++)
      results.push_back(workers_.schedule(&RandArray::construct, this,i,smallProblemThres));
    for(int i=0; i< numN_/smallProblemThres; i++) results[i].get();
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
  void construct(const int frame, const int taskRange);
  void recBitonicSort(const int lo, const int cnt, const int direct, const int ID);
  void sortFinalize(const int cnt, const int ID);
  void bitonicMerge(const int lo, const int cnt, const int direct, const int ID);
  void sortContin(const int lo, const int cnt, const int dir, const int ID);
  void mergeContin(const int lo, const int cnt, const int dir, const int ID,
      const int prereqStart, const int prereqEnd);
  void mergeFinalize(const int cnt, const int ID);
  inline void exchange(const int a, const int b) {
    int tmp;
    tmp=data_[a], data_[a]=data_[b], data_[b]=tmp;
  }

  const int threadN_, numN_;
  //! Size of array slice for each thread
  ThreadPool& workers_;
  unique_ptr<int[]> data_;
  ConcurMap taskComplete_, exchangeComplete_;
  atomic_int serial_;
  static const int seqThres_, ASCENDING, DESCENDING;
  //! Signal that all tasks have finished
  std::mutex finishMut_;
  std::condition_variable finishCnd_;
  bool finished_;
};
const int RandArray::seqThres_= 1<<0, RandArray::ASCENDING=1, RandArray::DESCENDING=0;

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

int compUP (const void *a, const void *b) {return ( *(int*)a - *(int*)b );}
int compDN (const void *a, const void *b) {return ( *(int*)b - *(int*)a );}

void RandArray::construct(const int frame, const int taskRange){
  const int start= frame*taskRange, end= (frame+1)*taskRange;
  // Hopefully the C++ stdlib implementation of rand() has no data races, unlike the C version
  // As mentioned here: http://www.cplusplus.com/reference/cstdlib/rand/
  for(int i=start; i<end; i++) data_[i]= rand() %20;
}
void RandArray::sort(){
  finished_= false;
  COUT<<"Scheduling tasks...\n";
  recBitonicSort(0,numN_,ASCENDING,0);
  COUT << "All tasks scheduled!\n";
  std::unique_lock<std::mutex> lk(finishMut_);
  finishCnd_.wait(lk, [=] { return finished_; });
  COUT << "Waited as well\n";
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

// Only insert prereq if this is a left-node (worker==true)
void RandArray::recBitonicSort(const int lo, const int cnt, const int dir, const int ID){
  if (cnt>seqThres_) {
    DBG_PRINTF("[recBitonicSort]: recursing\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    int k=cnt/2;
    workers_.schedule(&RandArray::recBitonicSort, this,lo, k, ASCENDING, ID+1);
    recBitonicSort(lo+k, k, DESCENDING, ID+cnt);
    workers_.schedule([=] (){
        workers_.schedule(&RandArray::sortContin,this,lo,cnt,dir,ID);
    });
  } else{
    DBG_PRINTF("[recBitonicSort]: LEAF\t\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    if(dir) qsort(data_.get()+lo, cnt, sizeof(int),compUP);
    else qsort(data_.get()+lo, cnt, sizeof(int),compDN);
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
void RandArray::sortContin(const int lo, const int cnt, const int dir, const int ID){
  ConcurMap::const_accessor ac;
  if(taskComplete_.find(ac, ID+1))
    if(ac->second == 2){
      ac.release();
      if(taskComplete_.find(ac, ID+cnt))
        if(ac->second == 2){
          ac.release();
          DBG_PRINTF("[sortContin]: continuing\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
          bitonicMerge(lo,cnt,dir, ID);
          workers_.schedule(&RandArray::sortFinalize, this, cnt, ID);
          return;
        }
    }
  //If dependency isn't complete, reschedule
  DBG_PRINTF("[sortContin]: rescheduling\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  workers_.schedule(&RandArray::sortContin, this,lo,cnt,dir,ID);
}
//! Signal this task is complete and erase its dependencies, which are no longer needed 
void RandArray::sortFinalize(const int cnt, const int ID){
  ConcurMap::const_accessor ac;
  if(taskComplete_.find(ac, ID))
    if(ac->second == 1){
      ac.release();
      DBG_PRINTF("[sortFinalize]: Making 2\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
      ConcurMap::accessor acMod;
      if(taskComplete_.find(acMod, ID)) acMod->second= 2;
      else throw new domain_error("[sortFinalizing]: ERROR! Current node #"+
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
  DBG_PRINTF("[sortFinalize]: rescheduling\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  workers_.schedule(&RandArray::sortFinalize, this, cnt, ID);
}

//! For small problems, synchronously merge; for larger sizes, launch asynchronous merging tasks
void RandArray::bitonicMerge(const int lo, const int cnt, const int dir, const int ID){
  if (cnt>1) {
    DBG_PRINTF("[bitonicMerge]: Enter\t\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
    int k= cnt>>1;
    const int smallProblemThres= (seqThres_<<0 < k)? seqThres_<<0: k;
    if (smallProblemThres < k){
      const int chunkNumber= k/smallProblemThres;
      // Request a range of serial numbers
      const int serialStart= serial_.fetch_add(chunkNumber);
      const int serialEnd= serialStart+chunkNumber;
      for(int i=0; i< chunkNumber; i++){
        workers_.schedule([=] (const int serial){
          const int start= lo+i*smallProblemThres, end= lo+(i+1)*smallProblemThres;
          for(int i=start; i<end; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k);
          this->exchangeComplete_.insert(make_pair(serial,1));
        }, serialStart+i);
      }
      // Schedule the rest of bitonicMerge
      workers_.schedule(&RandArray::mergeContin, this,lo,cnt,dir,ID, serialStart,serialEnd);
    } else {  // If problem is too small, run everything sequentially
      DBG_PRINTF("[bitonicMerge]: Leaf\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
      for (int i=lo; i<lo+k; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k);
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

void RandArray::mergeContin(const int lo, const int cnt, const int dir, const int ID,
    const int prereqStart, const int prereqEnd){
  ConcurMap::const_accessor ac;
  for(int serial= prereqStart; serial< prereqEnd; serial++){
    if(!exchangeComplete_.find(ac, serial)){
      // TODO: Schedule-to-reschedule? (perf)
      workers_.schedule(&RandArray::mergeContin, this,lo,cnt,dir, ID, prereqStart, prereqEnd);
      return;
    }
    ac.release();
  }
  // All prerequisites have completed!
  DBG_PRINTF("[mergeContin]: Schedul_merges\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  const int k= cnt>>1;
  workers_.schedule(&RandArray::bitonicMerge, this, lo,k,dir, ID+1);
  bitonicMerge(lo+k, k, dir, ID+cnt);
  // Schedule a rescheduling of mergeFinalize (which signals merge completion for this ID)
  workers_.schedule([=] (){
    workers_.schedule(&RandArray::mergeFinalize, this, cnt, ID);
  });
}
// After previous bitonic merges have completed, signal completion
void RandArray::mergeFinalize(const int cnt, const int ID){
  ConcurMap::const_accessor ac;
  if(taskComplete_.find(ac, ID+1))
    if(ac->second == 1){
      ac.release();
      if(taskComplete_.find(ac, ID+cnt))
        if(ac->second == 1){
          ac.release();
          DBG_PRINTF("[mergeFinalize]: Making 1\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
          if(!taskComplete_.insert(make_pair(ID,1))){
            ConcurMap::accessor acMod;
            if(taskComplete_.find(acMod, ID)) acMod->second= 1;
            else throw new domain_error("[merge]: ERROR! Current node #"+
                                        to_string(ID)+" was JUST deleted by someone else!\n");
          }
          DBG_PRINTF("[mergeFinalize]: Sched_sortFin\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
          return;
        }
    }
  ac.release();
  //reschedule
  {
    if(taskComplete_.find(ac, ID+1)) DBG_PRINTF("[mergeFin]: ID+1 status=%d\t#%d\t%zu\n", ac->second, ID, hash<thread::id>()(this_thread::get_id())%(1<<10));

    ac.release();
    if(taskComplete_.find(ac, ID+cnt)) DBG_PRINTF("[mergeFin]: ID+cnt status=%d\t#%d\t%zu\n", ac->second, ID, hash<thread::id>()(this_thread::get_id())%(1<<10));

    ac.release();
  }
  DBG_PRINTF("[mergeFinalize]: Reschedule\t#%d\t%zu\n", ID, hash<thread::id>()(this_thread::get_id())%(1<<10));
  workers_.schedule(&RandArray::mergeFinalize, this, cnt, ID);
}





