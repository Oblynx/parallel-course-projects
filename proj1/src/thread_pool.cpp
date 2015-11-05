#include "thread_pool.h"
#include <iostream>
#include <chrono>

void ThreadPool::startWorkers()
{
  // continue only if !beginworking
  if (beginworking_.test_and_set()) return;
  
  std::cout << "[ThreadPool]: Starting workers!\n";
  stop_= false;
  if(!threadNum_) throw std::invalid_argument("[CudaService]: More than zero threads expected");
  workers_.reserve(threadNum_);
  for(; threadNum_; --threadNum_)
    workers_.emplace_back([this] (){
      while(true)
      {
        std::function<void()> task;
        { // BEGIN Critical section
          std::unique_lock<std::mutex> lock(queue_mutex_);
          condition_.wait(lock,
                               [this]{ return stop_ || !tasks_.empty(); });
          std::cout << "[worker]: Woke up\n";
          if(!stop_){
            std::cout << "[worker]: Pulling up the sleeves...\n";
            task= std::move(tasks_.front());
            tasks_.pop();
          }
        } // END Critical section
        if(stop_){
          std::cout << "[worker#"<<std::this_thread::get_id()<<"]: Exit\n";
          break;
        }
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
  std::cout << "[ThreadPool]: Stopping workers. Size: " << workers_.size()<< "\n";
  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout << workers_.size() << '\n';
  for(auto&& worker: workers_){
    std::cout << "\t-->joining\n";
    worker.join();
    std::cout << "\t   joined <--\n";
  }
  std::cout << "[ThreadPool]: Clearing...\n";
  //workers_.clear();
  
  std::cout << "[ThreadPool]: Stopped workers!\n";
  beginworking_.clear();
}
