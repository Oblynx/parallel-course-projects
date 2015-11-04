#include "thread_pool.h"

void ThreadPool::startWorkers()
{
  // continue only if !beginworking
  if (beginworking_.test_and_set()) return;

  stop_= false;
  if(!threadNum_) throw std::invalid_argument("[CudaService]: More than zero threads expected");
  workers_.reserve(threadNum_);
  for(; threadNum_; --threadNum_)
    workers_.emplace_back([this] (){
      while(!stop_)
      {
        std::function<void()> task;
        { // BEGIN Critical section
          std::unique_lock<std::mutex> lock(queue_mutex_);
          condition_.wait(lock,
                               [this]{ return stop_ || !tasks_.empty(); });
          if(stop_ && tasks_.empty())
            return;
          task = std::move(tasks_.front());
          tasks_.pop();
        } // END Critical section
        task();
      }
    });
  endworking_.clear();
}

void ThreadPool::stopWorkers()
{
  // continue only if !endworking
  if (endworking_.test_and_set()) return;

  stop_= true;
  condition_.notify_all();
  for(std::thread& worker: workers_)
    worker.join();
  workers_.clear();
  beginworking_.clear();
}
