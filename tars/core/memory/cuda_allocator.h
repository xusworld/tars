#pragma once

#include "tars/core/allocator.h"

namespace tars {

class CudaAllocator : public Allocator {
 public:
  CudaAllocator() = default;
  virtual ~CudaAllocator() = default;

  virtual Status allocate(void **ptr, const int32_t bytes) override;

  virtual Status release(void *ptr) override;

  virtual RuntimeType runtime_type() const override {
    return RuntimeType::CUDA;
  }

  virtual Status reset(void *ptr, const int32_t val,
                       const int32_t size) override;

  static CudaAllocator *get();

 private:
  static CudaAllocator *alloc_;
};

}  // namespace tars