#include "glog/logging.h"
#include "tars/core/memory/memory_pool.h"

namespace ace {

template <RuntimeType rtype>
Status MemoryPool<rtype>::acuqire(void** ptr, const int32_t bytes) {
  auto iter = free_blocks_.lower_bound(bytes);
  if (iter == free_blocks_.end()) {
    CHECK_EQ(allocator_->allocate(ptr, bytes), Status::OK())
        << "Malloc new memory failed, please check.";
    // mark as a used memory block
    used_blocks_.emplace(bytes, ptr);
  } else {
    *ptr = iter->second;
    used_blocks_.emplace(iter->first, iter->second);
    free_blocks_.erase(iter);
  }

  return Status::OK();
}

template <RuntimeType rtype>
Status MemoryPool<rtype>::release(void* ptr) {
  for (auto& block : used_blocks_) {
    if (block.second == ptr) {
      free_blocks_.emplace(block);
      used_blocks_.erase(block.first);
      return Status::OK();
    }
  }
  return Status::ERROR("can't find pointer in memory pool, please check.");
}

template <RuntimeType rtype>
Status MemoryPool<rtype>::release_all(bool is_free_memory) {
  if (is_free_memory) {
    return this->clear();
  }

  for (auto& block : used_blocks_) {
    free_blocks_.emplace(block);
  }
  used_blocks_.clear();

  return Status::UNIMPLEMENTED("release interface is not implemented yet.");
}

template <RuntimeType rtype>
Status MemoryPool<rtype>::clear() {
  for (auto& block : used_blocks_) {
    allocator_->release(block.second);
  }
  used_blocks_.clear();

  for (auto& block : free_blocks_) {
    allocator_->release(block.second);
  }
  free_blocks_.clear();

  return Status::OK();
}

}  // namespace ace