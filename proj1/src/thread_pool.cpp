#include "thread_pool.h"

void ThreadPool::startWorkers()
{
  // continue only if !beginworking
  if (beginworking_.test_and_set()) return;

  stop_= false;
  if(!threadNum_) throw std::invalid_argument("[CudaService]: More than zero threads expected");
  workers_.reserve(threadNum_);
  workerWaiting_= 0;
  for(unsigned i=0; i<threadNum_; i++){
    workers_.emplace_back([this,i] (){
      while(!stop_)
      {
        std::function<void()> task;
        try{
          workerWaiting_++;  //a.k.a. waiting
          tasks_.pop(task);
          workerWaiting_--;  //a.k.a. busy
          task();
        }catch(tbb::user_abort){
          // Normal control flow when the destructor is called
        }catch(...){
          // std::cout << "[CudaService]: Unhandled exception!\n";
          throw;
        }
      }
    });
  }
  endworking_.clear();
}

void ThreadPool::stopWorkers()
{
  // continue only if !endworking
  if (endworking_.test_and_set()) return;

  stop_= true;
  tasks_.abort();
  for(std::thread& worker: workers_)
    worker.join();
  workers_.clear();
  beginworking_.clear();
}
