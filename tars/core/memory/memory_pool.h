#pragma once

#include <map>
#include <unordered_map>

#include "tars/core/allocator.h"
#include "tars/core/memory/cpu_allocator.h"
#include "tars/core/memory/cuda_allocator.h"
#include "tars/core/status.h"
#include "tars/core/types.h"

namespace tars {

class MemoryManager {
 public:
  using Stream = void*;

  // acquire a buffer from memory pool
  static Status acuqire(void** ptr, const int32_t size) {
    return Status::UNIMPLEMENTED("acquire interface is not implemented yet.");
  }

  // release a buffer from memroy pool
  static Status release(void* ptr) {
    return Status::UNIMPLEMENTED("release interface is not implemented yet.");
  }

  // reset a buffer in memory pool
  static Status reset(void* ptr, const int32_t val, const int32_t size) {
    return Status::UNIMPLEMENTED("reset interface is not implemented yet.");
  }

  // memcpy
  static Status sync_memcpy(void* dst, size_t dst_offset, int dst_id,
                            const void* src, size_t src_offset, int src_id,
                            size_t count, MemcpyKind) {
    return Status::UNIMPLEMENTED(
        "sync_memcpy interface is not implemented yet.");
  }

  // asynchronize memcpy
  static Status async_memcpy(void* dst, size_t dst_offset, int dst_id,
                             const void* src, size_t src_offset, int src_id,
                             size_t count, MemcpyKind) {
    return Status::UNIMPLEMENTED(
        "async_memcpy interface is not implemented yet.");
  }

  //  memcpy peer to peer, for device memory copy between different devices
  static Status sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                                const void* src, size_t src_offset, int src_id,
                                size_t count) {
    return Status::UNIMPLEMENTED(
        "sync_memcpy_p2p interface is not implemented yet.");
  }

  // asynchronize memcpy peer to peer, for device memory copy between different
  // devices
  static Status async_memcpy_p2p(void* dst, size_t dst_offset, int dst_id,
                                 const void* src, size_t src_offset, int src_id,
                                 size_t count, Stream stream) {
    return Status::UNIMPLEMENTED(
        "async_memcpy_p2p interface is not implemented yet.");
  }
};

namespace {

using BlockList = std::multimap<int, void*>;

}

// TODO(xusworld):
// 1. 完善内存池设计
// 2. 优化内存池碎片
template <RuntimeType rtype = RuntimeType::CPU>
class MemoryPool {
 public:
  MemoryPool() {
    switch (rtype_) {
      case RuntimeType::CUDA:
        allocator_ = CudaAllocator::get();
        break;
      case RuntimeType::CPU:
        allocator_ = CpuAllocator::get();
        break;
      default:
        LOG(INFO) << "Unknown allocator type: " << RuntimeTypeToString(rtype)
                  << ", please check.";
        break;
    }
    rtype_ = allocator_->runtime_type();
    LOG(INFO) << "Create a memory pool for " << RuntimeTypeToString(rtype);
  }

  MemoryPool(Allocator* allocator) : allocator_(allocator) {
    rtype_ = allocator_->runtime_type();
    LOG(INFO) << "Create a memory pool for " << RuntimeTypeToString(rtype);
  }

  ~MemoryPool() {
    LOG(INFO) << "Release a memory pool for " << RuntimeTypeToString(rtype_);
    clear();
  }

  // acquire a buffer
  Status acuqire(void** ptr, const int32_t bytes);

  // release a buffer
  Status release(void* ptr);

  // release all buffer
  Status release_all(bool is_free_memory);

  // returns free memory blocks size
  int32_t free_blocks_num() const { return free_blocks_.size(); }

  // returns used memory blocks size
  int32_t used_blocks_num() const { return used_blocks_.size(); }

  // returns runtime type
  RuntimeType runtime_type() const { return rtype_; };

 private:
  Status clear();

 private:
  BlockList used_blocks_;
  BlockList free_blocks_;
  RuntimeType rtype_;
  Allocator* allocator_;
};

}  // namespace tars