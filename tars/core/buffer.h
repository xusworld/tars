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

namespace tars {

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

class Buffer {
 public:
  Buffer(const RuntimeType rtype = RuntimeType::CPU,
         const DataType dtype = DataType_DT_FLOAT, const int32_t size = 0)
      : rtype_(rtype), dtype_(dtype), shared_(false) {
    CHECK_GT(size, 0) << "buffer size: " << size
                      << " <= 0, which is not a valid size.";

    // allocate a new memory space, so buffer will be a unshared buffer
    DLOG(INFO) << "Buffer runtime type: " << RuntimeTypeToString(rtype);
    this->allocator_ = RuntimeTypeToAllocator(RuntimeType::CPU);
    this->device_id_ = this->allocator_->device_id();
    auto status = allocator_->allocate(&data_, size);
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

  template <typename T>
  operator const T*() const {
    return reinterpret_cast<const T*>(data_);
  }

  template <typename T>
  operator T*() {
    return reinterpret_cast<T*>(data_);
  }

  // returns const data pointer
  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(data_);
  }

  // returns data pointer
  template <typename T>
  T* mutable_data() const {
    return reinterpret_cast<T*>(data_);
  }

  // returns the device id
  int32_t device_id() const { return device_id_; }

  // returns buffer size
  int32_t size() const { return size_; }

  // returns buffer capacity
  int32_t capacity() const { return capacity_; }

  // checks whether this buffer is empty
  bool empty() const { return size_ == 0; }

  // checks whether this buffer is a shared buffer
  bool shared() const { return shared_; }

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

  // TODO(xusworld) impl this.
  void map() {}

  void unmap() {}

  void copy_host(const void* src, const size_t bytes) {
    this->clear();
    this->resize(bytes);

    if (rtype_ == RuntimeType::CPU) {
      // host to host memory copy
      ::memcpy(data_, src, bytes);
    } else if (rtype_ == RuntimeType::CUDA) {
      // host to device memory copy on default stream
      cuda::CopyH2D(data_, src, bytes, 0);
    }
  }

  void copy_host(cudaStream_t stream, const void* src, const size_t bytes) {
    this->clear();
    this->resize(bytes);

    if (rtype_ == RuntimeType::CPU) {
      // host to host memory copy
      ::memcpy(data_, src, bytes);
    } else if (rtype_ == RuntimeType::CUDA) {
      // host to device memory copy on non-default stream
      cuda::CopyH2D(data_, src, bytes, stream);
    }
  }

  void copy_device(const void* src, const size_t bytes) {
    if (rtype_ == RuntimeType::CPU) {
      // device to host memory copy on default stream
      cuda::CopyD2H(data_, src, bytes, 0);
    } else if (rtype_ == RuntimeType::CUDA) {
      // device to device memory copy on default stream
      cuda::CopyD2D(data_, src, bytes, 0);
    }
  }

  void copy_device(cudaStream_t stream, const void* src, const size_t bytes) {
    if (rtype_ == RuntimeType::CPU) {
      // device to host memory copy on non-default stream
      cuda::CopyD2H(data_, src, bytes, stream);
    } else if (rtype_ == RuntimeType::CUDA) {
      // device to deivce memory copy on non-default stream
      cuda::CopyD2D(data_, src, bytes, stream);
    }
  }

  // Resizes the container to contain count elements, does nothing if count ==
  // size().If the current size is greater than count, the container is reduced
  // to its first count elements.If the current size is less than count,1)
  // additional default-inserted elements are appended
  void resize(const size_t count) {
    if (size_ >= count) {
      size_ = count;
    } else if (capacity_ >= count) {
      size_ = count;
    } else {
      // reallocate memory space
      auto new_cap = 2 * capacity_;
      if (count > new_cap) new_cap = count;
      this->realloc(new_cap);
    }
  }

  Status realloc(const int32_t size);

  // Increase the capacity of the vector (the total number of elements that the
  // vector can hold without requiring reallocation) to a value that's greater
  // or equal to new_cap. If new_cap is greater than the current capacity(), new
  // storage is allocated, otherwise the function does nothing.
  void reserve(const size_t new_cap);

  // reset memory
  Status reset(const int32_t val);

  // // reserve a buffer
  // void reserve(size_t new_cap, cudaStream_t stream = 0) {
  //   if (new_cap > capacity_) {
  //     // reallocate(new_cap, stream);
  //   }
  // }

 private:
  void init() {
    allocator_ = RuntimeTypeToAllocator(rtype_);
    rtype_ = allocator_->runtime_type();
  }

  // buffer's device id, which device the buffer belongs to
  int32_t device_id_ = 0;

  // buffer's allocator
  Allocator* allocator_;
  // buffer's runtime type
  RuntimeType rtype_;
  // buffer's data type
  DataType dtype_;
  // buffer's underlying memory
  void* data_ = nullptr;
  // buffer's underlying memory
  void* mapped_data_ = nullptr;

  // buffer's bytes
  int32_t size_ = 0;
  // buffer's capacity
  int32_t capacity_ = 0;
  // a flag, mark data_ in the buffer shared with other buffer or not
  bool shared_ = false;
};

}  // namespace tars
