#pragma once

#include "tars/core/runtime.h"

namespace tars {
namespace device {

class CudaRuntime : public Runtime {
 public:
  using Event = cudaEvent_t;
  using Stream = cudaStream_t;

 public:
  CudaRuntime() : type_(RuntimeType::CUDA) {}
  ~CudaRuntime() = default;

  virtual Status acuqire(void**, const int32_t size);
  virtual Status release(void*);
  virtual Status reset(void* ptr, const int32_t val, const int32_t size);

  virtual Status sync_memcpy(void* dst, size_t dst_offset, int dst_id,
                             const void* src, size_t src_offset, int src_id,
                             size_t count, MemcpyKind);
  virtual Status async_memcpy(void* dst, size_t dst_offset, int dst_id,
                              const void* src, size_t src_offset, int src_id,
                              size_t count, MemcpyKind);

  //  memcpy peer to peer, for device memory copy between different devices
  static void sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                              const void* src, size_t src_offset, int src_id,
                              size_t count){};

  // asynchronize memcpy peer to peer, for device memory copy between different
  // devices
  static void async_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                               const void* src, size_t src_offset, int src_id,
                               size_t count, Stream stream){};

  virtual Status create_event(Event* event, bool flag = false);
  virtual Status destroy_event(Event event);
  virtual Status record_event(Event event, Stream stream);
  virtual Status query_event(Event event);
  virtual Status sync_event(Event event);

  virtual Status create_stream(Stream* stream);
  virtual Status create_stream_with_flag(Stream* stream, unsigned int flag);
  virtual Status create_stream_with_priority(Stream* stream, unsigned int flag,
                                             int priority);

  virtual Status destroy_stream(Stream stream);
  virtual Status sync_stream(Event event, Stream stream);
  virtual Status sync_stream(Stream stream);

 protected:
  RuntimeType type_;
};

}  // namespace device
}  // namespace tars