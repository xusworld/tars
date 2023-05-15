#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "tars/core/macro.h"
#include "tars/core/thread_safe_macros.h"
#include "tars/core/type_traits.h"

namespace ace {

class ThreadPool {
  using Task = std::function<void(void)>;
  using Worker = std::thread;

 public:
  ThreadPool(int num_thread) : num_thread_(num_thread) {}
  virtual ~ThreadPool();

  // Launch tasks.
  void launch();

  // Lanuch the normal function task in sync.
  template <typename functor, typename... ParamTypes>
  typename function_traits<functor>::return_type RunSync(functor function,
                                                         ParamTypes... args);

  // Lanuch the normal function task in async.
  template <typename functor, typename... ParamTypes>
  typename std::future<typename function_traits<functor>::return_type> RunAsync(
      functor function, ParamTypes... args);

  // Stop the pool.
  void stop();

 private:
  // The initial function should be overrided by user who derive the ThreadPool
  // class.
  virtual void init();

  // Auxiliary function should be overrided when you want to do other things in
  // the derived class.
  virtual void auxiliary_funcs();

 private:
  std::mutex mutex_;
  int32_t num_thread_;
  std::queue<Task> tasks_ GUARDED_BY(mutex_);
  std::vector<Worker> workers_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace ace