#include <glog/logging.h>
#include <mm_malloc.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "tars/core/memory/cpu_allocator.h"
#include "tars/core/status.h"
#include "tars/core/utils.h"

namespace ace {

#define X64_ALIGNED_BYTES 32

Status CpuAllocator::allocate(const DataType dtype, const int32_t size,
                              void **ptr) {
  const auto bytes = GetBufferBytes(dtype, size);
  *ptr = reinterpret_cast<void *>(_mm_malloc(bytes, X64_ALIGNED_BYTES));
  CHECK(*ptr != NULL) << "malloc return a null pointer, please check";
  return Status::OK();
}

Status CpuAllocator::allocate(const int32_t bytes, void **ptr) {
  *ptr = reinterpret_cast<void *>(_mm_malloc(bytes, X64_ALIGNED_BYTES));
  CHECK(*ptr != NULL) << "malloc return a null pointer, please check";
  return Status::OK();
}

Status CpuAllocator::release(void *ptr) {
  if (ptr != nullptr) {
    // Memory that is allocated using _mm_malloc must be freed using _mm_free.
    _mm_free(ptr);
  } else {
    LOG(INFO) << "ptr is a null pointer, please check.";
  }

  return Status::OK();
}

Status sync_memcpy(void *dst, size_t dst_offset, int dst_id, const void *src,
                   size_t src_offset, int src_id, size_t bytes,
                   MemcpyKind kind) {
  return Status::OK();
}

Status async_memcpy(void *dst, size_t dst_offset, int dst_id, const void *src,
                    size_t src_offset, int src_id, size_t bytes, MemcpyKind) {
  return Status::OK();
}

Status CpuAllocator::reset(const DataType dtype, const size_t val,
                           const int32_t bytes, void *ptr) {
  memset(ptr, val, bytes);

  return Status::OK();
}

CpuAllocator *CpuAllocator::get() {
  if (alloc_ == nullptr) {
    alloc_ = new CpuAllocator;
  }

  return alloc_;
}

CpuAllocator *CpuAllocator::alloc_ = nullptr;

}  // namespace ace