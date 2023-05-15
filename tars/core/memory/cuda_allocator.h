#pragma once

#include "tars/core/allocator.h"

namespace ace {

class CudaAllocator : public Allocator {
 public:
  CudaAllocator() = default;
  virtual ~CudaAllocator() = default;

  virtual Status allocate(const DataType dtype, const int32_t size,
                          void **ptr) override;

  virtual Status allocate(const int32_t bytes, void **ptr) override;

  virtual Status release(void *ptr) override;

  virtual RuntimeType runtime_type() const override {
    return RuntimeType::CUDA;
  }

  virtual Status reset(const DataType dtype, const size_t val,
                       const int32_t bytes, void *ptr) override;

  static CudaAllocator *get();

 private:
  static CudaAllocator *alloc_;
};

}  // namespace ace