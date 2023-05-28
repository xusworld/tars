#include "tars/device/cpu_runtime.h"

namespace tars {
namespace device {

const int MALLOC_ALIGN = 64;

static inline void* fast_malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  char* p = static_cast<char*>(malloc(offset + size));

  if (!p) {
    return nullptr;
  }

  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  memset(r, 0, size);
  return r;
}

static inline void fast_free(void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}

Status CpuRuntime::acuqire(void** ptr, const int32_t size) {
  *ptr = (void*)fast_malloc(size);
  return Status::OK();
}

Status CpuRuntime::release(void* ptr) {
  if (ptr != nullptr) {
    fast_free(ptr);
  }
  return Status::OK();
}

Status CpuRuntime::reset(void* ptr, const int32_t val, const int32_t size) {
  memset(ptr, val, size);
  return Status::OK();
}

Status CpuRuntime::sync_memcpy(void* dst, size_t dst_offset, int dst_id,
                               const void* src, size_t src_offset, int src_id,
                               size_t count, MemcpyKind) {
  memcpy((char*)dst + dst_offset, (char*)src + src_offset, count);
  return Status::OK();
}

Status CpuRuntime::async_memcpy(void* dst, size_t dst_offset, int dst_id,
                                const void* src, size_t src_offset, int src_id,
                                size_t count, MemcpyKind) {
  memcpy((char*)dst + dst_offset, (char*)src + src_offset, count);
  return Status::OK();
}

Status CpuRuntime::create_event(Event* event, bool flag) {
  return Status::OK();
}

Status CpuRuntime::destroy_event(Event event) { return Status::OK(); }

Status CpuRuntime::record_event(Event event, Stream stream) {
  return Status::OK();
}

Status CpuRuntime::query_event(Event event) { return Status::OK(); }

Status CpuRuntime::sync_event(Event event) { return Status::OK(); }

Status CpuRuntime::create_stream(Stream* stream) { return Status::OK(); }

Status CpuRuntime::create_stream_with_flag(Stream* stream, unsigned int flag) {
  return Status::OK();
}

Status CpuRuntime::create_stream_with_priority(Stream* stream,
                                               unsigned int flag,
                                               int priority) {
  return Status::OK();
}

Status CpuRuntime::destroy_stream(Stream stream) { return Status::OK(); }

Status CpuRuntime::sync_stream(Event event, Stream stream) {
  return Status::OK();
}

Status CpuRuntime::sync_stream(Stream stream) { return Status::OK(); }

}  // namespace device
}  // namespace tars