#include "tars/core/macro.h"
#include "tars/core/status.h"
#include "tars/device/runtime/cuda_runtime.h"

namespace tars {
namespace device {

Status CudaRuntime::acuqire(void** ptr, const int32_t size) {
  CUDA_CHECK(cudaMallocHost(ptr, size));
  return Status::OK();
}

Status CudaRuntime::release(void* ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
  return Status::OK();
}

Status CudaRuntime::reset(void* ptr, const int32_t val, const int32_t size) {
  memset(ptr, val, size);
  return Status::OK();
}

Status CudaRuntime::sync_memcpy(void* dst, size_t dst_offset, int dst_id,
                                const void* src, size_t src_offset, int src_id,
                                size_t count, MemcpyKind) {
  CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                        cudaMemcpyHostToHost));
  CUDA_CHECK(cudaStreamSynchronize(0));
  //LOG(INFO) << "NVH, sync, H2H, size: " << count << ", src_offset: " \
          << src_offset << ", data:" << ((const float*)((char*)src + src_offset))[0];
  return Status::OK();
}

Status CudaRuntime::async_memcpy(void* dst, size_t dst_offset, int dst_id,
                                 const void* src, size_t src_offset, int src_id,
                                 size_t count, MemcpyKind) {
  CUDA_CHECK(cudaMemcpy((char*)dst + dst_offset, (char*)src + src_offset, count,
                        cudaMemcpyHostToHost));
  // LOG(INFO) << "NVH, sync, H2H, size: " << count;
  return Status::OK();
}

Status CudaRuntime::create_event(Event* event, bool flag) {
  if (flag) {
    CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDefault));
  } else {
    CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
  }
  return Status::OK();
}

Status CudaRuntime::destroy_event(Event event) {
  CUDA_CHECK(cudaEventDestroy(event));
  return Status::OK();
}

Status CudaRuntime::record_event(Event event, Stream stream) {
  CUDA_CHECK(cudaEventRecord(event, stream));
  return Status::OK();
}

Status CudaRuntime::query_event(Event event) {
  CUDA_CHECK(cudaEventQuery(event));
  return Status::OK();
}

Status CudaRuntime::sync_event(Event event) {
  CUDA_CHECK(cudaEventSynchronize(event));
  return Status::OK();
}

Status CudaRuntime::create_stream(Stream* stream) {
  CUDA_CHECK(cudaStreamCreate(stream));
  return Status::OK();
}

Status CudaRuntime::create_stream_with_flag(Stream* stream, unsigned int flag) {
  CUDA_CHECK(cudaStreamCreateWithFlags(stream, flag));
  return Status::OK();
}

Status CudaRuntime::create_stream_with_priority(Stream* stream,
                                                unsigned int flag,
                                                int priority) {
  CUDA_CHECK(cudaStreamCreateWithPriority(stream, flag, priority));
  return Status::OK();
}

Status CudaRuntime::destroy_stream(Stream stream) {
  CUDA_CHECK(cudaStreamDestroy(stream));
  return Status::OK();
}

Status CudaRuntime::sync_stream(Event event, Stream stream) {
  CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
  return Status::OK();
}

Status CudaRuntime::sync_stream(Stream stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::OK();
}
}  // namespace device
}  // namespace tars