#pragma once

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