#pragma once

#include "tars/core/memory/memory_pool.h"
#include "tars/core/runtime.h"
#include "tars/core/thread/thread_pool.h"
#include "tars/ir/types_generated.h"

namespace tars {
namespace device {

class CpuRuntime : public Runtime {
 public:
  using Event = void*;
  using Stream = void*;

 public:
  CpuRuntime() : type_(RuntimeType::CPU) {}
  ~CpuRuntime() = default;

  virtual Status acuqire(void**, const int32_t size) override;
  virtual Status release(void*) override;
  virtual Status reset(void* ptr, const int32_t val,
                       const int32_t size) override;

  virtual Status sync_memcpy(void* dst, size_t dst_offset, int dst_id,
                             const void* src, size_t src_offset, int src_id,
                             size_t count, MemcpyKind) override;
  virtual Status async_memcpy(void* dst, size_t dst_offset, int dst_id,
                              const void* src, size_t src_offset, int src_id,
                              size_t count, MemcpyKind) override;

  //  memcpy peer to peer, for device memory copy between different devices
  static void sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                              const void* src, size_t src_offset, int src_id,
                              size_t count){};

  // asynchronize memcpy peer to peer, for device memory copy between different
  // devices
  static void async_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                               const void* src, size_t src_offset, int src_id,
                               size_t count, Stream stream){};

  virtual Status create_event(Event* event, bool flag = false) override;
  virtual Status destroy_event(Event event) override;
  virtual Status record_event(Event event, Stream stream) override;
  virtual Status query_event(Event event) override;
  virtual Status sync_event(Event event) override;

  virtual Status create_stream(Stream* stream) override;
  virtual Status create_stream_with_flag(Stream* stream,
                                         unsigned int flag) override;
  virtual Status create_stream_with_priority(Stream* stream, unsigned int flag,
                                             int priority) override;

  virtual Status destroy_stream(Stream stream) override;
  virtual Status sync_stream(Event event, Stream stream) override;
  virtual Status sync_stream(Stream stream) override;

 protected:
  RuntimeType type_ = RuntimeType::CPU;
  MemoryPool<RuntimeType::CPU> memory_pool_;
  // ThreadPool thread_pool_;
};

}  // namespace device
}  // namespace tars