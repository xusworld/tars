#include <glog/logging.h>

#include "tars/core/buffer.h"
#include "tars/core/status.h"

namespace ace {

template <typename T>
Status Buffer<T>::reallocate(const int32_t size) {
  if (data_ != nullptr) {
    LOG(WARNING) << "Not a empty buffer, please check";
  }

  if (size_ <= 0) {
    LOG(ERROR) << "Try to allocate " << size_
               << " bytes memory space, which is < 0";
  }

  if (size > capacity_) {
    if (shared_) {
      LOG(INFO) << "shard buffer, can't realloc.";
    } else {
      auto ptr = reinterpret_cast<void *>(data_);
      this->allocator_->allocate(dtype_, size, &ptr);
      if (ptr == nullptr) {
        LOG(ERROR) << "Allocate memory failed.";
        return Status::FATAL("Allocate memory failed.");
      }
      capacity_ = size;
    }
  }
  size_ = size;
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
  if (new_size > capacity_) {
    size_t new_cap = 2 * capacity_;
    if (new_size > new_cap) new_cap = new_size;
    reallocate(new_cap);
  }
  size_ = new_size;
}

template class Buffer<float>;
template class Buffer<int>;

}  // namespace ace