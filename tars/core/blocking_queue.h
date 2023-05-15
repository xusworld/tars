#include "tars/core/macro.h"

namespace ace {

template <typename T>
class BlockingQueue {
 public:
  DISABLE_COPY_MOVE_ASSIGN(BlockingQueue);
  BlockingQueue() = delete;

  BlockingQueue(int32_t capacity) : capacity_(capacity), is_closed_(false) {}

  void Push(const T& val) {
    std::unique_lock<std::mutex> lck(mtx_);
    push_cond_.wait(lck, [&]() {
      return capacity_ - q_.size() >= 1 || is_closed_ == true;
    });
    PL_CHECK_EQ(is_closed_, false);
    q_.push(val);
    pop_cond_.notify_one();
  }

  void Pop(T* val) {
    std::unique_lock<std::mutex> lck(mtx_);
    pop_cond_.wait(lck,
                   [&]() { return q_.empty() == false || is_closed_ == true; });
    PL_CHECK_EQ(is_closed_, false);
    *val = q_.front();
    q_.pop();
    push_cond_.notify_one();
  }

  void Close() {
    std::unique_lock<std::mutex> lck(mtx_);
    is_closed_ = true;
    push_cond_.notify_all();
    pop_cond_.notify_all();
  }

 private:
  std::mutex mtx_;
  std::condition_variable push_cond_;
  std::condition_variable pop_cond_;
  std::queue<T> q_;
  int32_t capacity_;
  bool is_closed_;
};

}  // namespace ace