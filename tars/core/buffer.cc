#include <glog/logging.h>

#include "tars/core/buffer.h"
#include "tars/core/status.h"

namespace tars {

Status Buffer::realloc(const int32_t size) {
  if (data_ != nullptr) {
    LOG(WARNING) << "try to realloc a non-empty buffer";
  }

  if (size <= 0) {
    LOG(ERROR) << "try to allocate a invalid buffer size, size: " << size;
    return Status::ERROR("a invalid buffer size");
  }

  if (size > capacity_) {
    if (shared_) {
      LOG(INFO) << "try to realloc a shared buffer, nothing will happen";
    } else {
      // allocator a new buffer
      this->allocator_->allocate(&data_, size);

      if (data_ == nullptr) {
        LOG(ERROR) << "allocate new memory failed";
        return Status::FATAL("allocate memory failed");
      }

      // update the value of size_ and capacity_
      bytes_ = size;
      capacity_ = size;
    }
  }
  return Status::OK();
}

void Buffer::reserve(const size_t new_cap) {
  if (new_cap > capacity_) {
    this->realloc(new_cap);
  }
}

Status Buffer::reset(const int32_t val) {
  CHECK(data_ != nullptr) << ", null pointer, cannot reset.";
  this->allocator_->reset(data_, val, bytes_);
  return Status::OK();
}

}  // namespace tars