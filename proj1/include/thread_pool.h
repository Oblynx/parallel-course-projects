#ifndef THREAD_POOL
#define THREAD_POOL

#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <functional>

/*! Maintains a vector of waiting threads, which consume tasks from a task queue.
    - Although a tbb::concurrent_bounded_queue is used by default, an std::queue could
    also be used if an external synchronization mechanism is provided (less efficient).
    - Also a tbb::concurrent_queue can be used, if an external mechanism for waiting
    before queue pop is provided (mutex + condition variable)
  */ 
  class ThreadPool{
  public:
    //! Default constructor.
    ThreadPool(int n):  threadNum_(n), stop_(false) {
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
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.emplace([task](){ (*task)(); });
      }
      condition_.notify_one();
      return resultFut;
    }

    //! Destructor stops workers, if they are still running.
    virtual ~ThreadPool(){
      stopWorkers();
    }

    //! Only for testing. [release]->remove
    void setWorkerN(const int& n) { threadNum_= n; }
  protected:
    //! Constructs workers and sets them waiting. [release]->protected
    void startWorkers();
    //! Joins all worker threads. [release]->protected
    void stopWorkers();
    //! Threads that consume tasks
    std::vector< std::thread > workers_;
    //! Concurrent queue that produces tasks
    std::queue< std::function<void()> > tasks_;
    size_t threadNum_= 0;
  private:
    //! workers_ finalization flag
    std::atomic_bool stop_;
    //! Synchronize queue
    std::mutex queue_mutex_;
    std::condition_variable condition_;
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
