#include "dot_product.h"

#if defined(__GNUC__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif

namespace ace {
namespace device {
namespace x86 {

static inline float horizontal_add(__m256& x) {
  __m128 y = _mm256_extractf128_ps(x, 1);
  y = _mm_add_ps(_mm256_castps256_ps128(x), y);
  y = _mm_hadd_ps(y, y);
  y = _mm_hadd_ps(y, y);
  return _mm_cvtss_f32(y);
}

float _avx2_dot_product(const float* x, const float* y, int n) {
  _mm256_zeroupper();
  constexpr int kPack = 8;
  constexpr int kUnroll = 8;

  n /= kPack;

  __m256 p[kUnroll];

  for (int i = 0; i < kUnroll; ++i) p[i] = _mm256_xor_ps(p[i], p[i]);

  __m256 px, py;

  for (int i = 0; i < n / kUnroll; ++i) {
    for (int k = 0; k < kUnroll; ++k) {
      px = _mm256_load_ps(x + k * kPack);
      py = _mm256_load_ps(y + k * kPack);
      p[k] = _mm256_fmadd_ps(px, py, p[k]);
    }
    x += kUnroll * kPack;
    y += kUnroll * kPack;
  }

  int nleft = n - (n / kUnroll) * kUnroll;

  // handle left part, i.e. 32 element
  if (nleft > 0) {
    for (int k = 0; k < nleft; ++k) {
      px = _mm256_load_ps(x + k * kPack);
      py = _mm256_load_ps(y + k * kPack);
      p[k] = _mm256_fmadd_ps(px, py, p[k]);
    }
  }

  int m = kUnroll;
  while (m > 1) {
    m /= 2;
    for (int k = 0; k < m; ++k) {
      p[k] = _mm256_add_ps(p[k], p[k + m]);
    }
  }

  float result = horizontal_add(p[0]);
  _mm256_zeroupper();

  return result;
}

}  // namespace x86
}  // namespace device
}  // namespace ace