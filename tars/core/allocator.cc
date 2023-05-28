#include <cstdint>

#include "tars/core/allocator.h"
#include "tars/core/runtime.h"

namespace tars {

int32_t Allocator::device_id() const { return 0; }

}  // namespace tars