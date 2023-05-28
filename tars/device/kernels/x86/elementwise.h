#pragma once

#include <iostream>
#include <vector>

namespace tars {
namespace device {
namespace x86 {

void _avx2_abs(float* outputs, float* inputs, const int32_t size);

void _avx2_bounded_relu(float* outputs, float* inputs, const int32_t size,
                        const float threshold);

void _avx2_clip(float* outputs, const float* inputs, const int32_t size,
                const float min, const float max);

void _avx2_clipv2(float* outputs, float* inputs, const int32_t size,
                  const float min, const float max);

void _avx2_clipped_relu(float* outputs, float* inputs, const int32_t size,
                        const float threshold);

void _avx2_sin(float* outputs, float* inputs, const int32_t size);

void _avx2_cos(float* outputs, float* inputs, const int32_t size);

void _avx2_tan(float* outputs, float* inputs, const int32_t size);

void _avx2_elu(float* outputs, float* inputs, const int32_t size);

void _avx2_exp(float* outputs, float* inputs, const int32_t size);

void _avx2_exp2(float* outputs, float* inputs, const int32_t size);

void _avx2_floor(float* outputs, float* inputs, const int32_t size);

void _avx2_gelu(float* outputs, float* inputs, const int32_t size);

void _avx2_hard_sigmoid(float* outputs, float* inputs, const int32_t size,
                        const float alpha, const float beta);

void _avx2_hard_swish(float* outputs, float* inputs, const int32_t size,
                      const float shift, const float scale);

void _avx2_leaky_relu(float* outputs, float* inputs, const int32_t size,
                      const float alpha);

void _avx2_log(float* outputs, float* inputs, const int32_t size);

void _avx2_logistic(float* outputs, float* inputs, const int32_t size);

void _avx2_log_sigmoid(float* outputs, float* inputs, const int32_t size);

void _avx2_mish(float* outputs, float* inputs, const int32_t size,
                const float threshold);

void _avx2_power(float* outputs, float* inputs, const int32_t size,
                 const int32_t n);

void _avx2_prelu(float* outputs, float* inputs, const int32_t size,
                 const float alpha);

void _avx2_relu(float* outputs, float* inputs, const int32_t size);

void _avx2_relu6(float* outputs, float* inputs, const int32_t size);

void _avx2_round(float* outputs, float* inputs, const int32_t size,
                 const float min, const float max);

void _avx2_selu(float* outputs, float* inputs, const int32_t size);

void _avx2_square(float* outputs, float* inputs, const int32_t size);

void _avx2_sigmoid(float* outputs, float* inputs, const int32_t size);

void _avx2_soft_relu(float* outputs, float* inputs, const int32_t size,
                     const float td = 40.0f);

void _avx2_sqrt(float* outputs, float* inputs, const int32_t size);

void _avx2_swish(float* outputs, float* inputs, const int32_t size);

void _avx2_tanh(float* outputs, float* inputs, const int32_t size,
                const float min, const float max);

}  // namespace x86
}  // namespace device
}  // namespace tars