//! @file Parallel bitonic sort implemented with C++11 std threads
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
using namespace std;

class RandArray{
  public:
  //! Call workers to initialize random array
  RandArray(int threadN, int numN): threadN_(threadN), numN_(numN), threadRange_(numN/threadN),
      extraElts_(numN%threadN), data_(new int[numN]){//, checkCpy_(new int[numN]){
    srand(1);
    vector<thread> threads;
    threads.reserve(threadN);
    for(int i=0; i<threadN; i++) threads.emplace_back(&RandArray::construct,this,i);
    for(int i=0; i<threadN; i++) threads[i].join();
  }
  void sort();
  //! Check result correctness. Could also be a simple out-of-order search of course
  int check();

  private:
  //! Thread callback for creating random array slice
  void construct(int frame);
  int threadN_, numN_;
  //! Size of array slice for each thread [,+1]
  int threadRange_;
  //! How many threads still need to grab 1 extra element
  atomic<int> extraElts_;
  //! Data + copy for independent sorting to compare times
  unique_ptr<int[]> data_;//, checkCpy_;
};

int main(int argc, char** argv){
  if (argc<3){
    cout << "Parallel bitonic sort using STD threads.\n\
            Arguments:\n\t<log2 num of elements>\n\t<log2 num of threads>\n\n";
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
  threadN= 1<<logThreadN;
  RandArray array(threadN, numN);
  array.sort();
  return array.check();
}

//! CAS loop based on answer by Mike Vine: http://stackoverflow.com/questions/16870030
void RandArray::construct(int frame){
  bool extraElt= 0;
  int extraCompare[2]= {extraElts_,0};
  // Attempt to get an extra element
  while(true){
    if (extraCompare[0] <= 0) break;     // No more extra elts left
    extraCompare[1]= extraCompare[0]-1;  // Decrement common indicator if this thread gets elt
    // If compare succeeds, no other thread has intervened and this can safely take extra elt
    // Else, another thread was faster and this one takes the loop again
    if (extraElts_.compare_exchange_strong(extraCompare[0], extraCompare[1])){
      extraElt= 1;
      break;
    }
  }
  const int start= frame*threadRange_, end= (frame+1)*threadRange_+extraElt;
  // Hopefully the C++ stdlib implementation of rand() has no data races, unlike the C version
  // As mentioned here: http://www.cplusplus.com/reference/cstdlib/rand/
  for(int i=start; i<end; i++) data_[i]= rand();//, checkCpy_[i]= data_[i];
}

void RandArray::sort(){
}
//int compare (const void * a, const void * b) {return ( *(int*)a - *(int*)b );}
int RandArray::check(){
//  qsort(checkCpy_.get(), numN_, sizeof(int), compare);
  for(int i=0; i<numN_-1; i++) if(data_[i] > data_[i+1]) return false;
  return true;
}







