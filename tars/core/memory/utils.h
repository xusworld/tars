#pragma once

#include <cstdint>

#include "tars/core/macro.h"

namespace tars {
namespace cuda {

void CopyD2D(void* dst, const void* src, size_t n, cudaStream_t stream);

void CopyD2H(void* dst, const void* src, size_t n, cudaStream_t stream);

void CopyH2D(void* dst, const void* src, size_t n, cudaStream_t stream);

void CopyH2H(void* dst, const void* src, size_t n, cudaStream_t stream);

}  // namespace cuda
}  // namespace tars