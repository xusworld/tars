#include "softmax.h"

#if defined(__GNUC__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif

#include "avx2_funcs.h"

namespace tars {
namespace device {
namespace x86 {

void _avx2_softmax(float* outputs, const float* inputs, const int32_t size) {
  float tmpfloat8[8];
  int count = size / 8;
  int remain = count * 8;
  // step 1: get maxValue
  float maxValue = inputs[0];
  if (count > 0) {
    auto maxVal = _mm256_loadu_ps(inputs);
    for (int i = 1; i < count; i++) {
      maxVal = _mm256_max_ps(maxVal, _mm256_loadu_ps(inputs + i * 8));
    }
    _mm256_storeu_ps(tmpfloat8, maxVal);
    maxValue = tmpfloat8[0] > tmpfloat8[1] ? tmpfloat8[0] : tmpfloat8[1];
    for (int i = 2; i < 8; i++) {
      maxValue = maxValue > tmpfloat8[i] ? maxValue : tmpfloat8[i];
    }
  }
  for (int i = remain; i < size; i++) {
    maxValue = maxValue > inputs[i] ? maxValue : inputs[i];
  }

  // step 2: get exp(x - maxValue) and sum(exp(x - maxValue))
  float sumValue = 0.f;
  if (count > 0) {
    auto sumVal = _mm256_set1_ps(0.f);
    auto p0 = _mm256_set1_ps(0.6931471805599453);
    auto p1 = _mm256_set1_ps(1.4426950408889634);
    auto p2 = _mm256_set1_ps(1.f);
    auto p3 = _mm256_set1_ps(1.f);
    auto p4 = _mm256_set1_ps(0.5);
    auto p5 = _mm256_set1_ps(0.1666666666666666);
    auto p6 = _mm256_set1_ps(0.041666666666666664);
    auto p7 = _mm256_set1_ps(0.008333333333333333);
    auto xMax = _mm256_set1_ps(87);
    auto xMin = _mm256_set1_ps(-87);
    auto basic = _mm256_set1_epi32(1 << 23);
    auto temp127 = _mm256_set1_epi32(127);
    for (int i = 0; i < count; ++i) {
      auto x = _mm256_sub_ps(_mm256_loadu_ps(inputs + i * 8),
                             _mm256_set1_ps(maxValue));
      x = _mm256_max_ps(x, xMin);
      x = _mm256_min_ps(x, xMax);
      auto div = _mm256_mul_ps(x, p1);
      auto divInt = _mm256_cvtps_epi32(div);
      div = _mm256_cvtepi32_ps(divInt);
      auto div2 = _mm256_add_epi32(divInt, temp127);
      div2 = _mm256_mullo_epi32(div2, basic);
      auto expBasic = _mm256_castsi256_ps(div2);
      auto xReamin = _mm256_sub_ps(x, _mm256_mul_ps(div, p0));
      auto t = xReamin;
      auto c0 = _mm256_mul_ps(p7, t);
      auto c1 = _mm256_add_ps(c0, p6);
      auto c2 = _mm256_mul_ps(c1, t);
      auto c3 = _mm256_add_ps(c2, p5);
      auto c4 = _mm256_mul_ps(c3, t);
      auto c5 = _mm256_add_ps(c4, p4);
      auto c6 = _mm256_mul_ps(c5, t);
      auto c7 = _mm256_add_ps(c6, p3);
      auto c8 = _mm256_mul_ps(c7, t);
      auto c9 = _mm256_add_ps(c8, p2);
      auto expRemain = c9;
      auto expRes = _mm256_mul_ps(expBasic, expRemain);
      sumVal = _mm256_add_ps(expRes, sumVal);
      _mm256_storeu_ps(outputs + 8 * i, expRes);
    }
    _mm256_storeu_ps(tmpfloat8, sumVal);
    for (int i = 0; i < 8; i++) {
      sumValue += tmpfloat8[i];
    }
  }
  auto param = 0.6931471805599453;
  float xLimit = 87;
  for (int i = remain; i < size; i++) {
    auto x = inputs[i] - maxValue;
    x = x > -xLimit ? x : -xLimit;
    x = x < xLimit ? x : xLimit;

    int div = (x / param);
    int div2 = (div + 127) << 23;
    auto xReamin = x - div * param;
    float expBasic = *(float*)(&div2);

    auto t = xReamin;
    auto expRemain =
        ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t +
         1.0f) *
            t +
        1.0f;
    outputs[i] = expBasic * expRemain;
    sumValue += outputs[i];
  }
  // step 3: get x / sum and store
  for (int i = 0; i < count; ++i) {
    // using  1 / ((1 / x) * sum) instead x * (1 / sum) or x / sum for some
    // bugs in intel cpu
    auto x = _mm256_rcp_ps(_mm256_loadu_ps(outputs + 8 * i));
    auto y = _mm256_set1_ps(sumValue);
    auto z = _mm256_rcp_ps(_mm256_mul_ps(x, y));
    _mm256_storeu_ps(outputs + 8 * i, z);
  }
  sumValue = 1.f / sumValue;
  for (int i = remain; i < size; i++) {
    outputs[i] *= sumValue;
  }
}

/*
void avx2_vector_softmax(const float* in, int length, float* out) {
  float max = _m256_max_array(in, length);
  __m256 max_vec = _mm256_set1_ps(max);
  __m256 exp_sum = _mm256_setzero_ps();
  int remainder = length % 8;
  int round_length = length / 8 * 8;

  if (remainder > 0) {
    for (int j = 0; j < round_length; j += 8) {
      __m256 temp_in = _mm256_loadu_ps(&in[j]);
      __m256 temp_exp = exp256_ps_fma(temp_in - max_vec);
      exp_sum += temp_exp;
      _mm256_storeu_ps(&out[j], temp_exp);
    }

    __m256i vec_mask = _m256_continue_mask_m256i(remainder);
    __m256 vec_mask_m256 = _m256_continue_mask_m256(remainder);
    __m256 temp_in = _mm256_maskload_ps(&in[round_length], vec_mask);
    __m256 temp_exp = _mm256_blendv_ps(
        _mm256_setzero_ps(), exp256_ps_fma(temp_in - max_vec), vec_mask_m256);

    _mm256_maskstore_ps(&out[round_length], vec_mask, temp_exp);
    exp_sum += temp_exp;

    float sum = _m256_self_sum(exp_sum);
    __m256 sum_vec = _mm256_set1_ps(1.f / sum);

    for (int j = 0; j < round_length; j += 8) {
      __m256 temp_in = _mm256_loadu_ps(&out[j]);
      _mm256_storeu_ps(&out[j], temp_in * sum_vec);
    }

    temp_in = _mm256_maskload_ps(&out[round_length], vec_mask);
    _mm256_maskstore_ps(&out[round_length], vec_mask, temp_in * sum_vec);

  } else {
    for (int j = 0; j < round_length; j += 8) {
      __m256 temp_in = _mm256_loadu_ps(&in[j]);
      __m256 temp_exp = exp256_ps_fma(temp_in - max_vec);
      exp_sum += temp_exp;
      _mm256_storeu_ps(&out[j], temp_exp);
    }

    float sum = _m256_self_sum(exp_sum);
    __m256 sum_vec = _mm256_set1_ps(1.f / sum);

    for (int j = 0; j < round_length; j += 8) {
      __m256 temp_in = _mm256_loadu_ps(&out[j]);
      _mm256_storeu_ps(&out[j], temp_in * sum_vec);
    }
  }
}

void avx2_sequence_softmax(const float* data, std::vector<int>& seq_offset,
                           float* out) {
  for (int i = 0; i < seq_offset.size() - 1; i++) {
    int start = seq_offset[i];
    int end = seq_offset[i + 1];
    int length = end - start;
    const float* seq_in = &data[start];
    float* seq_out = &out[start];
    avx2_vector_softmax(seq_in, length, seq_out);
  }
}
*/

}  // namespace x86
}  // namespace device
}  // namespace tars