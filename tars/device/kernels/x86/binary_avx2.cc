#if defined(__GNUC__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif

#define VCL_NAMESPACE vector
#include "avx2_funcs.h"
#include "elementwise.h"
#include "vectorclass.h"
#include "vectormath_exp.h"
#include "vectormath_trig.h"

namespace ace {
namespace device {
namespace x86 {

void avx2_vector_sum(const float* in_0, const int len, float* out) {
  int round_dim = len / 8 * 8;
  int remainder = len % 8;
  __m256i mask_m256i = _m256_continue_mask_m256i(remainder);
#pragma omp parallel for schedule(static)

  for (int k = 0; k < round_dim; k += 8) {
    __m256 a = _mm256_loadu_ps(&in_0[k]);
    __m256 b = _mm256_loadu_ps(&out[k]);
    _mm256_storeu_ps(&out[k], _mm256_add_ps(a, b));
  }

  if (remainder > 0) {
    __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
    __m256 b = _mm256_maskload_ps(&out[round_dim], mask_m256i);
    _mm256_maskstore_ps(out + round_dim, mask_m256i, _mm256_add_ps(a, b));
  }
}

void avx2_vector_sum(const float* in_0, const float* in_1, const int len,
                     float* out) {
  int round_dim = len / 8 * 8;
  int remainder = len % 8;
  __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

  for (int k = 0; k < round_dim; k += 8) {
    __m256 a = _mm256_loadu_ps(&in_0[k]);
    __m256 b = _mm256_loadu_ps(&in_1[k]);
    _mm256_storeu_ps(&out[k], _mm256_add_ps(a, b));
  }

  if (remainder > 0) {
    __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
    __m256 b = _mm256_maskload_ps(&in_1[round_dim], mask_m256i);
    _mm256_maskstore_ps(out + round_dim, mask_m256i, _mm256_add_ps(a, b));
  }
}

void avx2_vector_sub(const float* in_0, const float* in_1, const int len,
                     float* out) {
  int round_dim = len / 8 * 8;
  int remainder = len % 8;
  __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

  for (int k = 0; k < round_dim; k += 8) {
    __m256 a = _mm256_loadu_ps(&in_0[k]);
    __m256 b = _mm256_loadu_ps(&in_1[k]);
    _mm256_storeu_ps(&out[k], _mm256_sub_ps(a, b));
  }

  if (remainder > 0) {
    __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
    __m256 b = _mm256_maskload_ps(&in_1[round_dim], mask_m256i);
    _mm256_maskstore_ps(out + round_dim, mask_m256i, _mm256_sub_ps(a, b));
  }
}

void avx2_vector_mul(const float* in_0, const float* in_1, const int len,
                     float* out) {
  int round_dim = len / 8 * 8;
  int remainder = len % 8;
  __m256i mask_m256i = _m256_continue_mask_m256i(remainder);

  for (int k = 0; k < round_dim; k += 8) {
    __m256 a = _mm256_loadu_ps(&in_0[k]);
    __m256 b = _mm256_loadu_ps(&in_1[k]);
    _mm256_storeu_ps(&out[k], _mm256_mul_ps(a, b));
  }

  if (remainder > 0) {
    __m256 a = _mm256_maskload_ps(&in_0[round_dim], mask_m256i);
    __m256 b = _mm256_maskload_ps(&in_1[round_dim], mask_m256i);
    _mm256_maskstore_ps(out + round_dim, mask_m256i, _mm256_mul_ps(a, b));
  }
}

}  // namespace x86
}  // namespace device
}  // namespace ace