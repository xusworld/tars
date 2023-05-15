#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "tars/core/allocator.h"
#include "tars/core/memory/cpu_allocator.h"
#include "tars/core/memory/cuda_allocator.h"
#include "tars/core/memory/utils.h"
#include "tars/core/runtime.h"
#include "tars/core/status.h"
#include "tars/core/tensor_shape.h"
#include "tars/core/type_traits.h"
#include "tars/core/types.h"
#include "tars/ir/op_option_generated.h"
#include "tars/ir/types_generated.h"

namespace ace {

namespace {

Allocator* RuntimeTypeToAllocator(const RuntimeType type) {
  switch (type) {
    case RuntimeType::CPU:
      return CpuAllocator::get();
    case RuntimeType::CUDA:
      return CudaAllocator::get();
    default:
      LOG(FATAL) << "failed.";
  }
}

}  // namespace

template <typename T>
class Buffer {
 public:
  Buffer(const RuntimeType rtype = RuntimeType::CPU,
         const DataType dtype = float32,  // use float as default data type
         const int32_t size = 0)
      : rtype_(rtype), dtype_(dtype), shared_(false) {
    CHECK_GT(size, 0) << "buffer size: " << size
                      << " <= 0, which is not a valid size.";

    // allocate a new memory space, so buffer will be a unshared buffer
    this->allocator_ = RuntimeTypeToAllocator(rtype);
    this->device_id_ = this->allocator_->device_id();
    auto status = this->reallocate(size);
    LOG(INFO) << "init a new buffer";
  }

  Buffer(const Buffer& other) {
    CHECK_GT(other.size_, 0) << "buffer size: " << other.size_
                             << " <= 0, which is not a valid size.";

    size_ = other.size_;
    // buffers on the same device
    if (this->allocator_->device_id() == other.device_id_) {
      data_ = other.data_;
      shared_ = other.shared_;
      capacity_ = other.capacity_;
    } else {
      LOG(FATAL) << "UNIMPLEMENTED!";
    }
  }

  Buffer(Buffer&& other) { *this = other; }

  Buffer& operator=(Buffer&& other) {
    rtype_ = other.rtype_;
    dtype_ = other.dtype_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    shared_ = false;
    data_ = other.data_;

    // set input buffer object to a empty buffer
    other.size_ = 0;
    other.capacity_ = 0;
    other.shared_ = false;
    return *this;
  }

  Buffer& operator=(Buffer& buf) {
    this->size_ = buf.size_;
    // buffer on the same device id
    if (this->device_id_ == buf.device_id_) {
      this->data_ = buf.data_;
      this->capacity_ = buf.capacity_;
      this->shared_ = false;
    } else {
      LOG(FATAL) << "UNIMPLEMENTED!";
    }
    return *this;
  }

  ~Buffer() {
    // may not free the memory space
    free();
  }

  // operator overloading
  operator T*() { return data_; }
  operator const T*() const { return data_; }

  // returns buffer's data pointer
  const T* data() const { return data_; }
  T* mutable_data() const { return data_; }

  // checks whether the container is empty
  bool empty() const { return size_ == 0; }

  // returns the number of elements
  int32_t size() const { return size_; }

  // returns bytes of the buffer
  int32_t bytes() const { return size_ * sizeof(T); }

  // returns the number of elements that can be held in currently allocated
  // storage
  int32_t capacity() const { return capacity_; };

  // clears the contents of the buffer
  void clear() { size_ = 0; }

  // free a buffer
  void free() {
    if (!shared_) {
      this->allocator_->release(data_);
    }

    size_ = 0;
    capacity_ = 0;
  }

#ifdef ACE_USE_CUDA
  // copy from host to device
  void from_host(const T* src, size_t count, cudaStream_t stream = 0) {
    clear();
    resize(count);
    cuda::copyH2D(data_, src, size(), stream);
  }

  // copy from device to device
  void from_device(const T* src, size_t count, cudaStream_t stream = 0) {
    clear();
    resize(count);
    cuda::copyD2D(data_, src, size(), stream);
  }

  // copy from device to device
  void copy(const Buffer& src, cudaStream_t stream = 0) {
    clear();
    resize(src.size());
    copyD2D(data_, src.data(), size(), stream);
  }

  // reserve a buffer
  void reserve(size_t new_cap, cudaStream_t stream = 0) {
    if (new_cap > capacity_) {
      // reallocate(new_cap, stream);
    }
  }

  // resize a buffer
  void resize(size_t new_size, cudaStream_t stream = 0) {
    if (new_size > capacity_) {
      size_t new_cap = 2 * capacity_;
      if (new_size > new_cap) new_cap = new_size;
      reallocate(new_cap, stream);
    }
    size_ = new_size;
  }
#endif

  // host memory copy
  void copy(const Buffer& src);

  // host memory reserve
  void reserve(size_t new_cap);

  // host memory resize
  void resize(const std::vector<int32_t>& dims);
  void resize(size_t size);

  // reallocte memory
  Status reallocate(const int32_t size);
  Status reallocate(const std::vector<int32_t>& dims);

  // reset memory
  Status reset(const T val);

 private:
  void init() {
    allocator_ = RuntimeTypeToAllocator(rtype_);
    rtype_ = allocator_->runtime_type();
  }

  // buffer's allocator
  Allocator* allocator_;
  // buffer's runtime type
  RuntimeType rtype_;
  // buffer's data type
  DataType dtype_;
  // buffer's elements number
  int32_t size_ = 0;
  // buffer's capacity, the number of elements that can be held in currently
  // allocated storage
  int32_t capacity_ = 0;
  // buffer's underlying memory
  T* data_ = nullptr;
  // buffer's device id, which device the buffer belongs to
  int32_t device_id_ = 0;
  // a flag, mark data_ in the buffer shared with other buffer or not
  bool shared_ = false;
};

}  // namespace ace
