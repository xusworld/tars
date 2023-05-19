#if defined(__GNUC__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif

#define VCL_NAMESPACE vector
#include "elementwise.h"
#include "vectorclass.h"
#include "vectormath_exp.h"
#include "vectormath_trig.h"

namespace ace {
namespace device {
namespace x86 {

void _avx2_abs(float* outputs, float* inputs, const int32_t size) {
  __m256 mask = _mm256_set_ps(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
                              0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    auto out = vector::Vec8f(_mm256_and_ps(x, mask));
    out.store(outputs + i);
  }
}

void _avx2_bounded_relu(float* outputs, float* inputs, const int32_t size,
                        const float threshold) {
  __m256 vthreshold = _mm256_set1_ps(threshold);
  __m256 zeros = _mm256_setzero_ps();
  int remainder = size % 8;
  int quotient = size / 8;

#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 y = _mm256_min_ps(_mm256_max_ps(x, zeros), vthreshold);
    _mm256_storeu_ps(outputs + i, y);
  }

  // if (remainder > 0) {
  //   __m256i vec_mask = _m256_continue_mask_m256i(remainder);
  //   __m256 temp = _mm256_maskload_ps(&in[quotient * 8], vec_mask);
  //   _mm256_maskstore_ps(&out[quotient * 8], vec_mask,
  //                       _mm256_max_ps(zeros, temp));
  // }
}

void _avx2_clip(float* outputs, const float* inputs, const int32_t size,
                const float min, const float max) {
  __m256 vmin = _mm256_set1_ps(min);
  __m256 vmax = _mm256_set1_ps(max);

#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    vector::Vec8f y = _mm256_max_ps(vmin, _mm256_min_ps(x, vmax));
    _mm256_store_ps(outputs + i, y);
  }
}

void _avx2_clipv2(float* outputs, float* inputs, const int32_t size,
                  const float min, const float max) {
  __m256 vmin = _mm256_set1_ps(min);
  __m256 vmax = _mm256_set1_ps(max);

#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    vector::Vec8f y = _mm256_max_ps(vmin, _mm256_min_ps(x, vmax));
    _mm256_store_ps(outputs + i, y);
  }
}

void _avx2_clipped_relu(float* outputs, float* inputs, const int32_t size,
                        const float threshold) {
  __m256 vthreshold = _mm256_set1_ps(threshold);
  __m256 zeros = _mm256_setzero_ps();
  int remainder = size % 8;
  int quotient = size / 8;

#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 y = _mm256_min_ps(_mm256_max_ps(x, zeros), vthreshold);
    _mm256_storeu_ps(outputs + i, y);
  }
}

void _avx2_sin(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    auto out = vector::sin(x);
    out.store(outputs + i);
  }
}

void _avx2_cos(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    auto out = vector::cos(x);
    out.store(outputs + i);
  }
}

void _avx2_tan(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    auto out = vector::tan(x);
    out.store(outputs + i);
  }
}

void _avx2_elu(float* outputs, float* inputs, const int32_t size) {
  __m256 alpha = _mm256_set1_ps(1.0f);

#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    __m256 exp = vector::exp(x);
    __m256 neg = _mm256_mul_ps(alpha, _mm256_sub_ps(exp, _mm256_set1_ps(1.0f)));
    __m256 mask = _mm256_cmp_ps(_mm256_setzero_ps(), x, _CMP_GT_OS);
    _mm256_storeu_ps(outputs + i, _mm256_blendv_ps(x, neg, mask));
  }
}

void _avx2_exp(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    __m256 y = vector::exp(x);
    _mm256_storeu_ps(outputs + i, y);
  }
}

void _avx2_exp2(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    __m256 y = vector::exp2(x);
    _mm256_storeu_ps(outputs + i, y);
  }
}

void _avx2_floor(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 y = _mm256_round_ps(x, 1 + 8);
    _mm256_storeu_ps(outputs + i, y);
  }
}

void _avx2_gelu(float* outputs, float* inputs, const int32_t size) {
  __m256 vc1 = _mm256_set1_ps(0.7978845608028654);
  __m256 vc2 = _mm256_set1_ps(0.044715);
  __m256 one = _mm256_set1_ps(1.0);
  __m256 half = _mm256_set1_ps(0.5);

#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f v = x;
    vector::Vec8f tmp = vector::mul_add(vc2, v * v * v, v) * vc1;
    vector::Vec8f y = v * half * (one + vector::tan(tmp));
    y.store(outputs + i);
  }
}

void _avx2_gelu_tanh(float* outputs, float* inputs, const int32_t size) {}

void _avx2_hard_sigmoid(float* outputs, float* inputs, const int32_t size,
                        const float alpha, const float beta) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 a = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(alpha), x),
                             _mm256_set1_ps(beta));
    __m256 y = _mm256_min_ps(_mm256_max_ps(_mm256_setzero_ps(), a),
                             _mm256_set1_ps(1.0f));
    _mm256_store_ps(outputs + i, y);
  }
}

void _avx2_hard_swish(float* outputs, float* inputs, const int32_t size,
                      const float shift, const float scale) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 maxval = _mm256_max_ps(
        _mm256_setzero_ps(),
        _mm256_min_ps(x, _mm256_set1_ps(shift)) + _mm256_set1_ps(shift));

    __m256 y = _mm256_mul_ps(_mm256_mul_ps(maxval, _mm256_set1_ps(scale)), x);
    _mm256_store_ps(outputs + i, y);
  }
}

void _avx2_leaky_relu(float* outputs, float* inputs, const int32_t size,
                      const float alpha) {
  __m256 m = _mm256_set1_ps(alpha);
#pragma omp parallel for
  for (unsigned i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 l = _mm256_mul_ps(x, m);
    __m256 y = _mm256_max_ps(x, l);
    _mm256_store_ps(outputs + i, y);
  }
}

void _avx2_log(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f exp = vector::log(x);
    exp.store(outputs + i);
  }
}

void _avx2_logistic(float* outputs, float* inputs, const int32_t size) {}

void _avx2_log_sigmoid(float* outputs, float* inputs, const int32_t size) {
  vector::Vec8f ones(1.0);
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f y = ones / (ones + vector::exp(-x));
    y.store(outputs + i);
  }
}

void _avx2_mish(float* outputs, float* inputs, const int32_t size,
                const float threshold) {}

void _avx2_power(float* outputs, float* inputs, const int32_t size,
                 const int32_t n) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f y = vector::pow(x, n);
    y.store(outputs + i);
  }
}

void _avx2_prelu(float* outputs, float* inputs, const int32_t size,
                 const float alpha) {
  __m256 valpha = _mm256_set1_ps(alpha);
  __m256 vzeros = _mm256_setzero_ps();
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 y = _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), x),
                             _mm256_mul_ps(valpha, _mm256_min_ps(vzeros, x)));
    _mm256_storeu_ps(outputs + i, y);
  }
}

void _avx2_relu(float* outputs, float* inputs, const int32_t size) {
  __m256 vzeros = _mm256_setzero_ps();

  int remainder = size % 8;
  int quotient = size / 8;

#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    __m256 y = _mm256_max_ps(x, vzeros);
    _mm256_storeu_ps(outputs + i, y);
  }
}

void _avx2_relu6(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 y = _mm256_max_ps(x, _mm256_set1_ps(6.0f));
    _mm256_storeu_ps(outputs + i, y);
  }
}

void _avx2_round(float* outputs, float* inputs, const int32_t size,
                 const float min, const float max) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    __m256 x = _mm256_load_ps(inputs + i);
    __m256 y = _mm256_round_ps(x, 0 + 8);
    _mm256_storeu_ps(outputs + i, y);
  }
}

void _avx2_selu(float* outputs, float* inputs, const int32_t size) {
  __m256 valpha = _mm256_set1_ps(1.673263);
  __m256 vscale = _mm256_set1_ps(1.05070098);
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f v1 = vscale * (valpha * vector::exp(x) - valpha);
    vector::Vec8f v2 = vscale * x;
    vector::Vec8f y = vector::max(v1, v2);
    y.store(outputs + i);
  }
}

void _avx2_square(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f y = vector::square(x);
    y.store(outputs + i);
  }
}

void _avx2_sigmoid(float* outputs, float* inputs, const int32_t size) {
  vector::Vec8f ones(1.0);
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f y = ones / (ones + vector::exp(x));
    y.store(outputs + i);
  }
}

// SoftRelu: out=ln(1+exp(max(min(x,threshold),−threshold)))
void _avx2_soft_relu(float* outputs, float* inputs, const int32_t size,
                     const float td) {
  vector::Vec8f threshold(td);
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f minval = vector::min(x, threshold);
    vector::Vec8f maxval = vector::max(minval, -threshold);
    vector::Vec8f y = vector::log(1 + vector::exp(maxval));
    y.store(outputs + i);
  }
}

void _avx2_sqrt(float* outputs, float* inputs, const int32_t size) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f y = vector::sqrt(x);
    y.store(outputs + i);
  }
}

void _avx2_swish(float* outputs, float* inputs, const int32_t size) {
  __m256 ones = _mm256_set1_ps(1.0);
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f denominator = ones + vector::sqrt(-x);
    vector::Vec8f y = x / denominator;
    y.store(outputs + i);
  }
}

void _avx2_tanh(float* outputs, float* inputs, const int32_t size,
                const float min, const float max) {
#pragma omp parallel for
  for (auto i = 0; i < size; i += 8) {
    vector::Vec8f x;
    x.load(inputs + i);
    vector::Vec8f y = vector::tan(x);
    y.store(outputs + i);
  }
}

}  // namespace x86
}  // namespace device
}  // namespace ace