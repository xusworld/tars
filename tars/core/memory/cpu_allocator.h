#pragma once

#include <cstdint>

#include "../allocator.h"

namespace tars {

class CpuAllocator : public Allocator {
 public:
  CpuAllocator() = default;
  virtual ~CpuAllocator() = default;

  virtual RuntimeType runtime_type() const override { return RuntimeType::CPU; }

  virtual Status allocate(void **ptr, const size_t bytes) override;

  virtual Status realloc(void **ptr, const size_t bytes) override;

  virtual Status reset(void *ptr, const int32_t val,
                       const size_t size) override;

  virtual Status release(void *ptr) override;

  static CpuAllocator *get();

 private:
  static CpuAllocator *alloc_;
};

}  // namespace tars