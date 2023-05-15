#pragma once

#include <glog/logging.h>

#include <cstdint>

#include "tars/core/status.h"
#include "tars/core/types.h"

namespace ace {

// Runtime Interface, a pure class.
// 1. memory
// 2. event
// 3. stream
class Runtime {
 public:
  using Event = void*;
  using Stream = void*;

  virtual Status acuqire(void**, const int32_t size) = 0;
  virtual Status release(void*) = 0;
  virtual Status reset(void* ptr, const int32_t val, const int32_t size) = 0;

  virtual Status sync_memcpy(void* dst, size_t dst_offset, int dst_id,
                             const void* src, size_t src_offset, int src_id,
                             size_t count, MemcpyKind) = 0;
  virtual Status async_memcpy(void* dst, size_t dst_offset, int dst_id,
                              const void* src, size_t src_offset, int src_id,
                              size_t count, MemcpyKind) = 0;

  //  memcpy peer to peer, for device memory copy between different devices
  static void sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                              const void* src, size_t src_offset, int src_id,
                              size_t count){};

  // asynchronize memcpy peer to peer, for device memory copy between different
  // devices
  static void async_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                               const void* src, size_t src_offset, int src_id,
                               size_t count, Stream stream){};

  virtual Status create_event(Event* event, bool flag = false) = 0;
  virtual Status destroy_event(Event event) = 0;
  virtual Status record_event(Event event, Stream stream) = 0;
  virtual Status query_event(Event event) = 0;
  virtual Status sync_event(Event event) = 0;

  virtual Status create_stream(Stream* stream) = 0;
  virtual Status create_stream_with_flag(Stream* stream, unsigned int flag) = 0;
  virtual Status create_stream_with_priority(Stream* stream, unsigned int flag,
                                             int priority) = 0;

  virtual Status destroy_stream(Stream stream) = 0;
  virtual Status sync_stream(Event event, Stream stream) = 0;
  virtual Status sync_stream(Stream stream) = 0;

 protected:
  RuntimeType type_;
};

}  // namespace ace