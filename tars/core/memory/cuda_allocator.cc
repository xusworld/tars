#include <cuda_runtime.h>
#include <glog/logging.h>
#include <mm_malloc.h>

#include <cstdint>
#include <cstdlib>

#include "tars/core/memory/cuda_allocator.h"

namespace ace {

Status CudaAllocator::allocate(const DataType dtype, const int32_t size,
                               void **ptr) {
  const auto bytes = GetBufferBytes(dtype, size);
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  *ptr = reinterpret_cast<char *>(ptr);

  CHECK(*ptr != NULL) << "malloc return a null pointer, please check";
  return Status::OK();
}

Status CudaAllocator::allocate(const int32_t bytes, void **ptr) {
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  *ptr = reinterpret_cast<char *>(ptr);

  CHECK(*ptr != NULL) << "malloc return a null pointer, please check";
  return Status::OK();
}

Status CudaAllocator::release(void *ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaFree(ptr));
  } else {
    LOG(INFO) << "ptr is a null pointer, please check.";
  }

  return Status::OK();
}

CudaAllocator *CudaAllocator::get() {
  if (alloc_ == nullptr) {
    alloc_ = new CudaAllocator;
  }

  return alloc_;
}

Status CudaAllocator::reset(const DataType dtype, const size_t val,
                            const int32_t bytes, void *ptr) {
  return Status::OK();
}

CudaAllocator *CudaAllocator::alloc_ = nullptr;

}  // namespace ace