#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "tars/core/runtime.h"
#include "tars/core/status.h"
#include "tars/core/tensor_shape.h"
#include "tars/core/types.h"

namespace tars {

// Allocator Inferface, a pure virtual class.
class Allocator {
 public:
  Allocator() = default;
  virtual ~Allocator() = default;

  // Every allocator always belongs to a device. For cpu allocator, device_id ==
  // 0; for cuda allocator, device_id == gpu_device_id
  virtual int32_t device_id() const;

  virtual Status allocate(void **ptr, const int32_t bytes) = 0;

  virtual Status release(void *ptr) = 0;

  virtual Status reset(void *ptr, const int32_t val, const int32_t size) = 0;

  virtual RuntimeType runtime_type() const = 0;
};

}  // namespace tars