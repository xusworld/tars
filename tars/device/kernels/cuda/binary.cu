#include "binary.cuh"


namespace tars {
namespace device {
namespace cuda {

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

__global__ void Relu(const float *input, float *output, size_t count,
                     float slope) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count);
       i += blockDim.x * gridDim.x) {
    float x = input[i];
    float y = x > 0 ? x : x * slope;
    output[i] = y;
  }
  return;
}

__global__ void Cast(float *input, float *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count);
       i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}

__global__ void CastMidFloat(float *input, float *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count);
       i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}

__global__ void CastBool(int32_t *input, int32_t *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count);
       i += blockDim.x * gridDim.x) {
    output[i] = input[i] > 0 ? 1 : 0;
  }
  return;
}

__global__ void Atan2(const float *input0, const float *input1, float *output,
                      size_t count, size_t s0, size_t s1) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count);
       i += blockDim.x * gridDim.x) {
    float x = input0[i * s0];
    float y = input1[i * s1];
    output[i] = atan2(x, y);
  }
  return;
}

__global__ void Mod(const float *input0, const float *input1, float *output,
                    size_t count, size_t s0, size_t s1) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count);
       i += blockDim.x * gridDim.x) {
    float x = input0[i * s0];
    float y = input1[i * s1];
    output[i] = x - x / y;
  }
  return;
}

__global__ void LogicalOR(const float *input0, const float *input1,
                          float *output, size_t count, size_t s0, size_t s1) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count);
       i += blockDim.x * gridDim.x) {
    float x = input0[i * s0];
    float y = input1[i * s1];
    output[i] = (x || y) ? 1 : 0;
  }
  return;
}

__global__ void Add(const int n, const float *in0, const float *in1, float *out,
                    int s0, int s1) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = in0[index * s0] + in1[index * s1]; }
}

__global__ void Sub(const int n, const float *in0, const float *in1, float *out,
                    int s0, int s1) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = in0[index * s0] - in1[index * s1]; }
}

__global__ void Mul(const int n, const float *in0, const float *in1, float *out,
                    int s0, int s1) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = in0[index * s0] * in1[index * s1]; }
}

__global__ void Div(const int n, const float *in0, const float *in1, float *out,
                    int s0, int s1) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = in0[index * s0] / in1[index * s1]; }
}

} // namespace binary
} // namespace cuda
} // namespace kernels