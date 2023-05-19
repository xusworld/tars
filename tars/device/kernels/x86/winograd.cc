#if defined(__GNUC__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif

#include "avx2_funcs.h"

namespace ace {
namespace device {
namespace x86 {

static void transpose(float* data_out, const float* data_in, int w_in,
                      int h_in) {
  for (int j = 0; j < h_in; ++j) {
    for (int i = 0; i < w_in; ++i) {
      data_out[i * h_in + j] = data_in[j * w_in + i];
    }
  }
}

static void winograd_transform_weights(float* dout, const float* din,
                                       int ch_out, int ch_in,
                                       float* work_space) {
  const float coeff[8][3] = {{1.0f, 0.0f, 0.0f},
                             {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                             {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                             {1.0f / 90, 1.0f / 45, 2.0f / 45},
                             {1.0f / 90, -1.0f / 45, 2.0f / 45},
                             {32.0f / 45, 16.0f / 45, 8.0f / 45},
                             {32.0f / 45, -16.0f / 45, 8.0f / 45},
                             {0.0f, 0.0f, 1.0f}};

  float* ptr_out = work_space;

  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const float* kernel0 = din + (i * ch_in + j) * 9;
      float* ptr_channel = ptr_out + (i * ch_in + j) * 64;

      //! transform kernel, transposed
      const float* k0 = kernel0;
      const float* k1 = kernel0 + 3;
      const float* k2 = kernel0 + 6;

      //! h
      float tmp[8][3];

      for (int i = 0; i < 8; i++) {
        tmp[i][0] =
            k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
        tmp[i][1] =
            k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
        tmp[i][2] =
            k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
      }

      //! v
      for (int j = 0; j < 8; j++) {
        float* tmpp = &tmp[j][0];

        for (int i = 0; i < 8; i++) {
          ptr_channel[j * 8 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
        }
      }
    }
  }

  transpose(static_cast<float*>(dout), ptr_out, 64, ch_out * ch_in);
}

static void winograd_transform_weights_oc_ic_64(float* dout, const float* din,
                                                int ch_out, int ch_in,
                                                float* work_space) {
  const float coeff[8][3] = {{1.0f, 0.0f, 0.0f},
                             {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                             {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                             {1.0f / 90, 1.0f / 45, 2.0f / 45},
                             {1.0f / 90, -1.0f / 45, 2.0f / 45},
                             {32.0f / 45, 16.0f / 45, 8.0f / 45},
                             {32.0f / 45, -16.0f / 45, 8.0f / 45},
                             {0.0f, 0.0f, 1.0f}};

  float* ptr_out = dout;

  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const float* kernel0 =
          static_cast<const float*>(din) + (i * ch_in + j) * 9;
      float* ptr_channel = ptr_out + (i * ch_in + j) * 64;

      //! transform kernel, transposed
      const float* k0 = kernel0;
      const float* k1 = kernel0 + 3;
      const float* k2 = kernel0 + 6;

      //! h
      float tmp[8][3];

      for (int i = 0; i < 8; i++) {
        tmp[i][0] =
            k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
        tmp[i][1] =
            k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
        tmp[i][2] =
            k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
      }

      //! v
      for (int j = 0; j < 8; j++) {
        float* tmpp = &tmp[j][0];

        for (int i = 0; i < 8; i++) {
          ptr_channel[j * 8 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
        }
      }
    }
  }
}

}  // namespace x86
}  // namespace device
}  // namespace ace
