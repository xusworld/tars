#include <cstdint>

#include "tars/core/allocator.h"
#include "tars/core/runtime.h"

namespace ace {

Status Allocator::allocate(const DataType dtype,
                           const std::vector<int32_t> &dims, void **ptr) {
  const auto size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int32_t>());
  return allocate(dtype, size, ptr);
}

Status Allocator::allocate(const DataType dtype, const TensorShape &shape,
                           void **ptr) {
  const auto size = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<int32_t>());
  return allocate(dtype, size, ptr);
}

int32_t Allocator::device_id() const { return 0; }

}  // namespace ace