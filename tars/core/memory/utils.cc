#include "tars/core/memory/utils.h"

namespace tars {
namespace cuda {

template <typename T>
void CopyD2D(T* dst, const T* src, size_t n, cudaStream_t stream) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice,
                            stream));
}

template <typename T>
void CopyD2H(T* dst, const T* src, size_t n, cudaStream_t stream) {
  CUDA_CALL(
      cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void CopyH2D(T* dst, const T* src, size_t n, cudaStream_t stream) {
  CUDA_CALL(
      cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <typename T>
void CopyH2H(T* dst, const T* src, size_t n, cudaStream_t stream) {
  CUDA_CALL(
      cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyHostToHost, stream));
}

}  // namespace cuda
}  // namespace tars