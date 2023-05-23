#pragma once

#include <cstddef>  // size_t

#include "ops/x64/blas/blas_info.h"

namespace ops {
namespace x64 {
namespace blas {

// y = x * W + b
// shape
//   x: [m]
//   W: [m, n]
//   b: [n]
//   y: [n]
void sgemv(float *y, const float *x, const float *W, MatrixLayout w_layout,
           const float *bias, int m, int n);
void sgemv_relu(float *y, const float *x, const float *W, MatrixLayout w_layout,
                const float *bias, int m, int n);

// W is column-major matrix
// Limitations:
//  1. m % 8 == 0
// support bias = nullptr (y = x * W) and y = bias (y += x * W)
void ref_sgemv_column_major(float *y, const float *x, const float *W,
                            const float *bias, int m, int n);
void sgemv_column_major(float *y, const float *x, const float *W,
                        const float *bias, int m, int n);

// multi-threading optimization
// Limitations:
//  1. m % 8 == 0
//  2. n % 4 == 0
// support bias = nullptr (y = x * W) and y = bias (y += x * W)
void sgemv_column_major_block(float *y, const float *x, const float *W,
                              const float *bias, int m, int n);

void sgemv_column_major_block_relu(float *y, const float *x, const float *W,
                                   const float *bias, int m, int n);

// W is row-major matrix
// Limitations:
//  1. n % 8 == 0
void ref_sgemv_row_major(float *y, const float *x, const float *W,
                         const float *bias, int m, int n);
void sgemv_row_major(float *y, const float *x, const float *W,
                     const float *bias, int m, int n);
void sgemv_row_major_relu(float *y, const float *x, const float *W,
                          const float *bias, int m, int n);

// The row-major multi-threading version need extra buffer
class RowMajorSgemv {
 public:
  size_t GetBufferSize(int n);

  void SetBuffer(const float *buffer) { buffer_ = const_cast<float *>(buffer); }

  void DoForward(float *y, const float *x, const float *W, const float *bias,
                 int m, int n);

 private:
  float *buffer_ = nullptr;
};

// row-major matrix, W in memory format: (n/64, m, 64) + (m, 8 * k), k = 0-7
void sgemv_row_major_w_n64(float *y, const float *x, const float *W,
                           const float *bias, int m, int n);

// column major matrix, W in memory format: (n/4 ,m/16, 4, 16)
void sgemv_column_major_w_m16_n4(float *y, const float *x, const float *W,
                                 const float *bias, int m, int n);

// column major matrix, W in memory format: (n/8 ,m/8, 8, 8)
void sgemv_column_major_w_m8_n8(float *y, const float *x, const float *W,
                                const float *bias, int m, int n);

}  // namespace blas
}  // namespace x64
}  // namespace ops
