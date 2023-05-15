#pragma once

#include <cstdint>

#include "../allocator.h"

namespace ace {

class CpuAllocator : public Allocator {
 public:
  CpuAllocator() = default;
  virtual ~CpuAllocator() = default;

  virtual Status allocate(const DataType dtype, const int32_t size,
                          void **ptr) override;

  virtual Status allocate(const int32_t bytes, void **ptr) override;

  virtual Status release(void *ptr) override;

  virtual Status reset(const DataType dtype, const size_t val,
                       const int32_t bytes, void *ptr) override;

  virtual RuntimeType runtime_type() const override { return RuntimeType::CPU; }

  static CpuAllocator *get();

 private:
  static CpuAllocator *alloc_;
};

}  // namespace ace