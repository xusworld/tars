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

  // returns the machine id
  virtual int32_t machine_id() const { return 0; }

  // returns the device id
  virtual int32_t device_id() const { return 0; }

  // returns the runtime type
  virtual RuntimeType runtime_type() const = 0;

  // allocates size bytes of uninitialized storage
  virtual Status allocate(void **ptr, const size_t size) = 0;

  // reallocates the given area of memory
  virtual Status realloc(void **ptr, const size_t new_size) = 0;

  // reset the given memory
  virtual Status reset(void *ptr, const int32_t val, const size_t size) = 0;

  // release the memory
  virtual Status release(void *ptr) = 0;
};

}  // namespace tars