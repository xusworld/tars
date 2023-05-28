#pragma once
#include <cuda_runtime.h>


namespace tars {
namespace device {
namespace cuda {

__global__ void Relu(const float *input, float *output, size_t count,
                     float slope);

__global__ void Cast(float *input, float *output, size_t count);

__global__ void CastMidFloat(float *input, float *output, size_t count);

__global__ void CastBool(int32_t *input, int32_t *output, size_t count);

__global__ void Atan2(const float *input0, const float *input1, float *output,
                      size_t count, size_t s0, size_t s1);

__global__ void Mod(const float *input0, const float *input1, float *output,
                    size_t count, size_t s0, size_t s1);

__global__ void LogicalOR(const float *input0, const float *input1,
                          float *output, size_t count, size_t s0, size_t s1);

__global__ void Add(const int n, const float *in0, const float *in1, float *out,
                    int s0, int s1);

__global__ void Sub(const int n, const float *in0, const float *in1, float *out,
                    int s0, int s1);

__global__ void Mul(const int n, const float *in0, const float *in1, float *out,
                    int s0, int s1);

__global__ void Div(const int n, const float *in0, const float *in1, float *out,
                    int s0, int s1);
} // namespace binary
} // namespace cuda
} // namespace kernels