#pragma once

#include <cstdint>

#include "tars/core/macro.h"

namespace tars {
namespace cuda {

template <typename T>
void CopyD2D(T* dst, const T* src, size_t n, cudaStream_t stream);

template <typename T>
void CopyD2H(T* dst, const T* src, size_t n, cudaStream_t stream);

template <typename T>
void CopyH2D(T* dst, const T* src, size_t n, cudaStream_t stream);

template <typename T>
void CopyH2H(T* dst, const T* src, size_t n, cudaStream_t stream);

}  // namespace cuda
}  // namespace tars