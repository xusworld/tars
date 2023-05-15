#pragma once

#include <cstdint>

#include "../allocator.h"

namespace ace {

class CpuAllocator : public Allocator {
 public:
  CpuAllocator() = default;
  virtual ~CpuAllocator() = default;

  virtual Status allocate(void **ptr, const int32_t bytes) override;

  virtual Status release(void *ptr) override;

  virtual Status reset(void *ptr, const int32_t val,
                       const int32_t size) override;

  virtual RuntimeType runtime_type() const override { return RuntimeType::CPU; }

  static CpuAllocator *get();

 private:
  static CpuAllocator *alloc_;
};

}  // namespace ace