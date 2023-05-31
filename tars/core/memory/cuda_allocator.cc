#include <cuda_runtime.h>
#include <glog/logging.h>
#include <mm_malloc.h>

#include <cstdint>
#include <cstdlib>

#include "tars/core/memory/cuda_allocator.h"

namespace tars {

Status CudaAllocator::allocate(void **ptr, const size_t bytes) {
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  *ptr = reinterpret_cast<char *>(ptr);

  CHECK(*ptr != NULL) << "malloc return a null pointer, please check";
  return Status::OK();
}

Status CudaAllocator::realloc(void **ptr, const size_t bytes) {
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

Status CudaAllocator::reset(void *ptr, const int32_t val, const size_t size) {
  return Status::OK();
}

CudaAllocator *CudaAllocator::alloc_ = nullptr;

}  // namespace tars