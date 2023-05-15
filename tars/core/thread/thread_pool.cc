#include "glog/logging.h"
#include "thread_pool.h"

namespace ace {

inline void ThreadPool::launch() {
  for (size_t i = 0; i < num_thread_; ++i) {
    workers_.emplace_back([i, this]() {
      // initial
      this->init();
      for (;;) {
        std::function<void(void)> task;
        {
          std::unique_lock<std::mutex> lock(this->mutex_);
          while (!this->stop_ && this->tasks_.empty()) {
            this->cv_.wait(lock);
          }
          if (this->stop_) {
            return;
          }
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }
        DLOG(INFO) << " Thread (" << i << ") processing";
        auxiliary_funcs();
        task();
      }
    });
  }
}

inline void ThreadPool::stop() {
  std::unique_lock<std::mutex> lock(this->mutex_);
  stop_ = true;
}

inline void ThreadPool::init() {
  LOG(INFO) << "Initialize a thread pool for test";
}

inline void ThreadPool::auxiliary_funcs() {}

inline ThreadPool::~ThreadPool() {
  stop();
  this->cv_.notify_all();
  for (auto& worker : workers_) {
    worker.join();
  }
}

template <typename functor, typename... ParamTypes>
inline typename function_traits<functor>::return_type ThreadPool::RunSync(
    functor function, ParamTypes... args) EXCLUSIVE_LOCKS_REQUIRED(_mut) {
  auto task = std::make_shared<std::packaged_task<
      typename function_traits<functor>::return_type(void)> >(
      std::bind(function, std::forward<ParamTypes>(args)...));
  std::future<typename function_traits<functor>::return_type> result =
      task->get_future();
  {
    std::unique_lock<std::mutex> lock(this->mutex_);
    this->tasks_.emplace([&]() { (*task)(); });
  }
  this->cv_.notify_one();
  return result.get();
}

template <typename functor, typename... ParamTypes>
inline std::future<typename function_traits<functor>::return_type>
ThreadPool::RunAsync(functor function, ParamTypes... args)
    EXCLUSIVE_LOCKS_REQUIRED(_mut) {
  auto task = std::make_shared<std::packaged_task<
      typename function_traits<functor>::return_type(void)> >(
      std::bind(function, std::forward<ParamTypes>(args)...));
  std::future<typename function_traits<functor>::return_type> result =
      task->get_future();
  {
    std::unique_lock<std::mutex> lock(this->mutex_);
    this->tasks_.emplace([=]() { (*task)(); });
  }
  this->cv_.notify_one();
  return result;
}
}  // namespace ace
