#include <omp.h>
#include <iostream>
#include <chrono>
#include <memory>

using namespace std;

int compUP (const void *a, const void *b) {return ( *(unsigned*)a - *(unsigned*)b );}
int compDN (const void *a, const void *b) {return ( *(unsigned*)b - *(unsigned*)a );}

class RandArray{
public:
  //! Call workers to initialize random array
  RandArray(const unsigned numN, const unsigned threadN): numN_(numN), threadN_(threadN),
      data_(new unsigned[numN]), seqThres_((numN/threadN)>>1){
    cout << "Constructing RandArray\n";
    #pragma omp parallel
    {
      unsigned seed= omp_get_thread_num();
      #pragma omp for schedule(static, seqThres_)
      for(unsigned i=0; i<numN; i++) data_[i]= rand_r(&seed) %2000;
    }
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
  void recBitonicSort(const unsigned lo, const unsigned cnt, const int direct);
  void bitonicMerge(const unsigned lo, const unsigned cnt, const int direct);
  inline void exchange(const unsigned a, const unsigned b) {
    unsigned tmp;
    tmp=data_[a], data_[a]=data_[b], data_[b]=tmp;
  }
  const unsigned numN_, threadN_;
  const unique_ptr<unsigned[]> data_;
  const unsigned seqThres_, ASCENDING=1, DESCENDING=0;
};

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
  omp_set_num_threads(threadN);
  auto start= chrono::system_clock::now();
  RandArray array(numN, threadN);
  chrono::duration<double> duration= chrono::system_clock::now()-start;
  cout<<"--> Array constructed in "<<duration.count()*1000<<"ms\n";                                 
  array.sort();
  duration= chrono::system_clock::now()-start;                                                      
  cout<<"--> Array sorted in "<<duration.count()*1000<<"ms\n";                                      
  return array.check();                                                                             
}    

void RandArray::sort(){
  #pragma omp parallel //num_threads(threadN_)
  #pragma omp single nowait
  recBitonicSort(0, numN_, ASCENDING);
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

void RandArray::recBitonicSort(const unsigned lo, const unsigned cnt, const int dir){
  if(cnt > seqThres_){
    unsigned k= cnt>>1;
    #pragma omp task
      recBitonicSort(lo, k, ASCENDING);
    recBitonicSort(lo+k, k, DESCENDING);
    #pragma omp taskwait
    bitonicMerge(lo, cnt, dir);
  } else {
    if(dir) qsort(data_.get()+lo, cnt, sizeof(unsigned),compUP);
    else    qsort(data_.get()+lo, cnt, sizeof(unsigned),compDN);
  }
}

void RandArray::bitonicMerge(const unsigned lo, const unsigned cnt, const int dir){
  const unsigned k= cnt>>1;
  if (k > seqThres_) {
    #pragma omp parallel for schedule(static, seqThres_) //num_threads(threadN_)
      for(unsigned i=lo; i<lo+k; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k); 
    #pragma omp task 
      bitonicMerge(lo, k, dir);
    bitonicMerge(lo+k, k, dir);
    #pragma omp taskwait
  } else if(k>0){
    for(unsigned i=lo; i<lo+k; i++) if(dir == (data_[i]>data_[i+k])) exchange(i,i+k); 
    bitonicMerge(lo, k, dir);
    bitonicMerge(lo+k, k, dir);
  }
}

