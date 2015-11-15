/*  The ThreadPool is based on a Github project. Notice:
  Copyright (c) 2012 Jakob Progsch, Václav Zeman
  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.
  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:
     1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
     2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
     3. This notice may not be removed or altered from any source
     distribution.

  --> This is an altered version of the original code by Konstantinos Samaras-Tsakiris.
*/
#ifndef THREAD_POOL
#define THREAD_POOL

#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <functional>

// Required to compile TBB properly with clang...
#define _LIBCPP_VERSION 1
// DEBUG: check the C++11 flags that enable "emplace"
//#include "tbb-4.4/tbb_config.h"
#include "tbb-4.4/concurrent_queue.h"

#include <iostream>

/*! Maintains a vector of waiting threads, which consume tasks from a task queue.
    - Although a tbb::concurrent_bounded_queue is used by default, an std::queue could
    also be used if an external synchronization mechanism is provided (less efficient).
    - Also a tbb::concurrent_queue can be used, if an external mechanism for waiting
    before queue pop is provided (mutex + condition variable)
  */ 
  class ThreadPool{
  public:
    //! Default constructor.
    ThreadPool(int n): threadNum_(n), stop_(false) {
      beginworking_.clear(); endworking_.test_and_set();
      startWorkers();
    }
    //!@{
    //! Copy and move constructors and assignments explicitly deleted.
    ThreadPool(const ThreadPool&) =delete;
    ThreadPool& operator=(const ThreadPool&) =delete;
    ThreadPool(ThreadPool&&) =delete;
    ThreadPool& operator=(ThreadPool&&) =delete;
    //!@}
    
    /*! Schedule a (cpu) task and get its future handle
      @param f Any callable object (function name, lambda, `std::function`, ...)
      @param args Variadic template parameter pack that holds all the arguments
      that should be forwarded to the callable object
      @return An `std::future` handle to the task's result
    */
    template<typename F, typename... Args>
    inline std::future<typename std::result_of<F(Args...)>::type>
      schedule(F&& f, Args&&... args)
    {
      using packaged_task_t = std::packaged_task<typename std::result_of<F(Args...)>::type ()>;

      std::shared_ptr<packaged_task_t> task(new packaged_task_t(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
      ));
      auto resultFut = task->get_future();
      tasks_.emplace([task](){ (*task)(); });
      return resultFut;
    }

    //! Clears tasks queue
    void clearTasks(){ tasks_.clear(); }

    // [[deprecated]]
    // void waitFinish(){
    //   while(!tasks_.empty() || workerWaiting_!=threadNum_ || !notTransientFinish_.test_and_set()){
    //     while(!tasks_.empty() || workerWaiting_!=threadNum_){
    //       std::this_thread::sleep_for(std::chrono::microseconds(threadNum_*20+50));
    //     }
    //     // FIXME!
    //     // Check that condition isn't transient. Certainly NOT a guarantee!!!
    //     notTransientFinish_.test_and_set();
    //     std::this_thread::sleep_for(std::chrono::microseconds(threadNum_*50));
    //   }
    // }
    
    //! Constructs workers and sets them waiting. [release]->protected
    void startWorkers();
    //! Joins all worker threads. [release]->protected
    void stopWorkers();
    //! Destructor stops workers, if they are still running.
    virtual ~ThreadPool(){
      stopWorkers();
    }

    //! Only for testing. [release]->remove
    void setWorkerN(const int& n) { threadNum_= n; }
    unsigned workers() { return threadNum_; }
  protected:
    //! Threads that consume tasks
    std::vector< std::thread > workers_;
    std::atomic<unsigned> workerWaiting_;
    //! Concurrent queue that produces tasks
    tbb::concurrent_bounded_queue< std::function<void()> > tasks_;
    size_t threadNum_= 0;
  private:
    //! workers_ finalization flag
    std::atomic_bool stop_;
    //!@{
    /*! Start/end workers synchronization flags.
      {beginworking_, endworking}: (initially: {F,T})
      1. {F,T}: not working
      2. {T,T}: transition
      3. {T,F}: working
    */
    std::atomic_flag beginworking_;
    std::atomic_flag endworking_;
    std::atomic_flag notTransientFinish_;
    //!@}
  };

#endif
