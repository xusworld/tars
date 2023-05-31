#include <glog/logging.h>
#include <mm_malloc.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "tars/core/memory/cpu_allocator.h"
#include "tars/core/status.h"
#include "tars/core/utils.h"

namespace tars {

#define X64_ALIGNED_BYTES 32

Status CpuAllocator::allocate(void **ptr, const size_t bytes) {
  *ptr = reinterpret_cast<void *>(::_mm_malloc(bytes, X64_ALIGNED_BYTES));
  CHECK(*ptr != NULL) << "malloc return a null pointer";
  return Status::OK();
}

Status CpuAllocator::realloc(void **ptr, const size_t bytes) {
  *ptr = ::realloc(ptr, bytes);
  CHECK(*ptr != NULL) << "realloc return a null pointer";
  return Status::OK();
}

Status CpuAllocator::release(void *ptr) {
  if (ptr != nullptr) {
    // memory that is allocated using _mm_malloc must be freed using _mm_free.
    _mm_free(ptr);
  } else {
    DLOG(INFO) << "ptr is a null pointer, no need to release again";
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

Status CpuAllocator::reset(void *ptr, const int32_t val, const size_t size) {
  ::memset(ptr, val, size);

  return Status::OK();
}

CpuAllocator *CpuAllocator::get() {
  if (alloc_ == nullptr) {
    alloc_ = new CpuAllocator;
  }

  return alloc_;
}

CpuAllocator *CpuAllocator::alloc_ = nullptr;

}  // namespace tars