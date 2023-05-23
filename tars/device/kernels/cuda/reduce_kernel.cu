#include <iostream>

#include "helper_cuda.h"
#include "reduce_kernel.h"

namespace kernels {
namespace cuda {

template <int32_t block_thread_num>
__device__ void ReduceSharedMemory(float* shm, float* result) {
  if (block_thread_num >= 1024) {
    if (threadIdx.x < 512) {
      shm[threadIdx.x] += shm[threadIdx.x + 512];
    }
    __syncthreads();
  }
  if (block_thread_num >= 512) {
    if (threadIdx.x < 256) {
      shm[threadIdx.x] += shm[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if (block_thread_num >= 256) {
    if (threadIdx.x < 128) {
      shm[threadIdx.x] += shm[threadIdx.x + 128];
    }
    __syncthreads();
  }
  if (block_thread_num >= 128) {
    if (threadIdx.x < 64) {
      shm[threadIdx.x] += shm[threadIdx.x + 64];
    }
    __syncthreads();
  }
  // the final warp
  if (threadIdx.x < 32) {
    volatile float* vshm = shm;
    if (blockDim.x >= 64) {
      vshm[threadIdx.x] += vshm[threadIdx.x + 32];
    }
    vshm[threadIdx.x] += vshm[threadIdx.x + 16];
    vshm[threadIdx.x] += vshm[threadIdx.x + 8];
    vshm[threadIdx.x] += vshm[threadIdx.x + 4];
    vshm[threadIdx.x] += vshm[threadIdx.x + 2];
    vshm[threadIdx.x] += vshm[threadIdx.x + 1];
    if (threadIdx.x == 0) {
      *result = vshm[0];
    }
  }
}

__device__ int32_t done_block_count = 0;

template <int32_t block_thread_num>
__global__ void SinglePassMergedKernel(const float* input, float* part_sum,
                                       float* output, size_t n) {
  int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int32_t total_thread_num = gridDim.x * blockDim.x;
  // printf("gridDim.x = %d, blockDim.x = %d\n", gridDim.x, blockDim.x);
  // reduce
  //   input[gtid + total_thread_num * 0]
  //   input[gtid + total_thread_num * 1]
  //   input[gtid + total_thread_num * 2]
  //   input[gtid + total_thread_num * ...]
  float sum = 0.0f;
  for (int32_t i = gtid; i < n; i += total_thread_num) {
    sum += input[i];
  }

  //  store sum to shared memory
  extern __shared__ float shm[];
  shm[threadIdx.x] = sum;
  __syncthreads();
  // reduce shared memory to part_sum
  ReduceSharedMemory<block_thread_num>(shm, part_sum + blockIdx.x);
  // make sure when a block get is_last_block is true,
  // all the other part_sums is ready
  __threadfence();
  // check if this block is the last
  __shared__ bool is_last_block;
  if (threadIdx.x == 0) {
    is_last_block = atomicAdd(&done_block_count, 1) == gridDim.x - 1;
  }
  __syncthreads();
  // reduce part_sum to output
  if (is_last_block) {
    sum = 0.0f;
    for (int32_t i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      sum += part_sum[i];
    }
    shm[threadIdx.x] = sum;
    __syncthreads();
    ReduceSharedMemory<block_thread_num>(shm, output);
    done_block_count = 0;
  }
}

void ReduceBySinglePass(const float* host_inputs, float* host_output,
                        size_t n) {
  const int32_t thread_num_per_block = 1024;
  const int32_t block_num = 1024;
  size_t shm_size = thread_num_per_block * sizeof(float);

  const size_t size_in_bytes = n * sizeof(float);
  // Allocate the device input vector A
  float* device_inputs = NULL;
  checkCudaErrors(cudaMalloc((void**)&device_inputs, size_in_bytes));
  float* device_partial_sum = NULL;
  checkCudaErrors(
      cudaMalloc((void**)&device_partial_sum, block_num * sizeof(float)));
  float* device_output = NULL;
  checkCudaErrors(cudaMalloc((void**)&device_output, sizeof(float)));

  // Initialize inputs
  cudaMemcpy(device_inputs, host_inputs, size_in_bytes, cudaMemcpyHostToDevice);
  cudaMemset(device_partial_sum, 0.0f, block_num * sizeof(float));
  cudaMemset(device_output, 0.0f, sizeof(float));

  KernelExecTimecost("ReduceBySinglePass", [&]() {
    SinglePassMergedKernel<thread_num_per_block>
        <<<block_num, thread_num_per_block, shm_size>>>(
            device_inputs, device_partial_sum, device_output, n);
  });

  cudaMemcpy(host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost);
  printf("%f\n", host_output[0]);
  // Deallocate device memory.
  cudaFree(device_inputs);
  cudaFree(device_partial_sum);
  cudaFree(device_output);
}

void KernelExecTimecost(const std::string& title,
                        std::function<void(void)> func) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  func();

  // record stop event on the default stream
  cudaEventRecord(stop);
  // wait until the stop event completes
  cudaEventSynchronize(stop);
  // calculate the elapsed time between two events
  float time;
  cudaEventElapsedTime(&time, start, stop);

  std::cout << title << " , timecost is " << time << " ms" << std::endl;

  // clean up the two events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

}  // namespace cuda
}  // namespace kernels