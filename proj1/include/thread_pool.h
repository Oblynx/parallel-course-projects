#ifndef THREAD_POOL
#define THREAD_POOL

#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <functional>
//Other tools
#include <tbb/concurrent_queue.h>
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
    void waitFinish(){
      while(!tasks_.empty() || workerWaiting_!=threadNum_){
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
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
    //!@}
  };

#endif
