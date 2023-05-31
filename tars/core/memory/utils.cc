#include "tars/core/memory/utils.h"

namespace tars {
namespace cuda {

void CopyD2D(void* dst, const void* src, size_t n, cudaStream_t stream) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n * sizeof(void),
                            cudaMemcpyDeviceToDevice, stream));
}

void CopyD2H(void* dst, const void* src, size_t n, cudaStream_t stream) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n * sizeof(void), cudaMemcpyDeviceToHost,
                            stream));
}

void CopyH2D(void* dst, const void* src, size_t n, cudaStream_t stream) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n * sizeof(void), cudaMemcpyHostToDevice,
                            stream));
}

void CopyH2H(void* dst, const void* src, size_t n, cudaStream_t stream) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n * sizeof(void), cudaMemcpyHostToHost,
                            stream));
}

}  // namespace cuda
}  // namespace tars