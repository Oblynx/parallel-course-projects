#include <omp.h>
#include <iostream>
#include <chrono>


using namespace std;

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
  auto start= chrono::system_clock::now();                                                          
  chrono::duration<double> duration= chrono::system_clock::now()-start;                             
  cout<<"--> Array constructed in "<<duration.count()*1000<<"ms\n";                                 
  duration= chrono::system_clock::now()-start;                                                      
  cout<<"--> Array sorted in "<<duration.count()*1000<<"ms\n";                                      
  return array.check();                                                                             
}    
