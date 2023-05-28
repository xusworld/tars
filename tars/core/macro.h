#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// TODO(xusworld) add a build macro
#include <cuda_runtime.h>

#include "tars/ir/types_generated.h"

// type alias, refer tars/ir/proto/types.fbs
// enum DataType : int8_t {
//   none = 0,
//   int8 = 1,
//   int16 = 2,
//   int32 = 3,
//   int64 = 4,
//   uint8 = 5,
//   uint16 = 6,
//   uint32 = 7,
//   uint64 = 8,
//   float16 = 9,
//   float32 = 10
// };

// Helper to delete copy constructor & copy-assignment operator
#define DISABLE_COPY_MOVE_ASSIGN(name)   \
  name(const name&) = delete;            \
  name& operator=(const name&) = delete; \
  name(name&&) = delete;                 \
  name& operator=(name&&) = delete

// Helper to declare copy constructor & copy-assignment operator default
#define DEFAULT_COPY_MOVE_ASSIGN(name)    \
  name(const name&) = default;            \
  name& operator=(const name&) = default; \
  name(name&&) = default;                 \
  name& operator=(name&&) = default

#define CUDA_NUM_THREADS 512

#define CUDA_KERNEL_LE(i, n)                     \
  int i = blockIdx.x * blockDim.x + threadIdx.x; \
  if (i < n)

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

/// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
inline int CUDA_GET_BLOCKS(const int N, const int base) {
  return (N + base - 1) / base;
}

#define CUDA_CALL(err, ...)                                                    \
  do {                                                                         \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA ERROR, line %d: %s: %s\n", __LINE__, cudaGetErrorName(err), \
             cudaGetErrorString(err));                                         \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

#define CUDA_CHECK(condition)                                         \
  do {                                                                \
    cudaError_t error = condition;                                    \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#ifdef USE_CUDNN
#include <cudnn.h>
#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition)                                               \
  do {                                                                       \
    cudnnStatus_t status = condition;                                        \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnn_get_errorstring(status); \
  } while (0)

const char* cudnn_get_errorstring(cudnnStatus_t status);
#endif  // USE_CUDNN