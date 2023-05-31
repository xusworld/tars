#pragma once

#include "tars/core/allocator.h"

namespace tars {

class CudaAllocator : public Allocator {
 public:
  CudaAllocator() = default;
  virtual ~CudaAllocator() = default;

  virtual RuntimeType runtime_type() const override {
    return RuntimeType::CUDA;
  }

  virtual Status allocate(void **ptr, const size_t bytes) override;

  virtual Status realloc(void **ptr, const size_t bytes) override;

  virtual Status reset(void *ptr, const int32_t val,
                       const size_t size) override;

  virtual Status release(void *ptr) override;

  static CudaAllocator *get();

 private:
  static CudaAllocator *alloc_;
};

}  // namespace tars