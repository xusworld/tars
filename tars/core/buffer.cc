#include <glog/logging.h>

#include "tars/core/buffer.h"
#include "tars/core/status.h"

namespace tars {

template <typename T>
Status Buffer<T>::reallocate(const int32_t size) {
  if (data_ != nullptr) {
    LOG(WARNING) << "Not a empty buffer, please check";
  }

  if (size <= 0) {
    LOG(ERROR) << "Try to allocate " << size
               << " bytes memory space, which is < 0";
  }

  LOG(INFO) << "size: " << size << " capacity: " << capacity_;
  if (size > capacity_) {
    if (shared_) {
      LOG(INFO) << "shard buffer, can't realloc.";
    } else {
      auto ptr = reinterpret_cast<void *>(data_);
      this->allocator_->allocate(&ptr, size * sizeof(T));
      data_ = reinterpret_cast<T *>(ptr);

      if (data_ == nullptr) {
        LOG(ERROR) << "Allocate memory failed.";
        return Status::FATAL("Allocate memory failed.");
      }
      // change the value of size_ and capacity_.
      size_ = size;
      capacity_ = size;
    }
  }
  return Status::OK();
}

template <typename T>
Status Buffer<T>::reallocate(const std::vector<int32_t> &dims) {
  const int32_t size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  return this->reallocate(size);
}

// host memory copy
template <typename T>
void Buffer<T>::copy(const Buffer<T> &src) {
  clear();
  resize(src.size());
  memcpy(data_, src.data(), size());
}

// host memory reserve
template <typename T>
void Buffer<T>::reserve(size_t new_cap) {
  if (new_cap > capacity_) {
    reallocate(new_cap);
  }
}

// host memory resize
template <typename T>
void Buffer<T>::resize(size_t new_size) {
  DLOG(INFO) << "buffer size: " << new_size << " , capacity: " << capacity_;
  if (new_size > capacity_) {
    size_t new_cap = 2 * capacity_;
    if (new_size > new_cap) new_cap = new_size;
    reallocate(new_cap);
  }
  size_ = new_size;
}
// host memory resize
template <typename T>
void Buffer<T>::resize(const std::vector<int32_t> &dims) {
  const int32_t size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  resize(size);
}

template <typename T>
Status Buffer<T>::reset(const T val) {
  CHECK(data_ != nullptr) << ", null pointer, cannot reset.";
  LOG(INFO) << "reset memory";
  memset(data_, val, size_ * sizeof(T));
  // this->allocator_->reset(data_, val, size_ * sizeof(T));
  return Status::OK();
}

template class Buffer<float>;
template class Buffer<int>;
template class Buffer<long>;

}  // namespace tars