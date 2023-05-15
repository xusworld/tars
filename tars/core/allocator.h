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

namespace ace {

// Allocator Inferface, a pure virtual class.
class Allocator {
 public:
  Allocator() = default;
  virtual ~Allocator() = default;

  // Every allocator always belongs to a device. For cpu allocator, device_id ==
  // 0; for cuda allocator, device_id == gpu_device_id
  virtual int32_t device_id() const;

  virtual Status allocate(const DataType dtype, const int32_t size,
                          void **ptr) = 0;

  virtual Status allocate(const DataType dtype,
                          const std::vector<int32_t> &dims, void **ptr);

  virtual Status allocate(const DataType dtype, const TensorShape &shape,
                          void **ptr);

  virtual Status allocate(const int32_t bytes, void **ptr) = 0;

  virtual Status release(void *ptr) = 0;

  virtual Status reset(const DataType dtype, const size_t val,
                       const int32_t bytes, void *ptr) = 0;

  virtual RuntimeType runtime_type() const = 0;
};

}  // namespace ace