#if defined(__GNUC__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif

#include <algorithm>
#include <cstring>

#include "sgemm.h"
#include "tars/core/utils.h"

namespace tars {
namespace device {
namespace x86 {

template <typename T>
inline T DivUp(T n, T div) {
  return (n + div - 1) / div * div;
}

void pack_no_trans_n6(float* a, const int lda, float* b, const int ldb,
                      const int m, const int n);
void pack_no_trans_nx(float* a, const int lda, float* b, const int ldb,
                      const int m, const int n, int cols);

//  pack block_size on non-leading dimension, n denotes no-transpose.
//  eg. input:   A MxN matrix in col major, so the storage-format is (N, M)
//      output:  B MxN matrix in col major(N-packed), so the storage-format is
//                    (divUp(N, block_size), M, block_size)
void pack_no_trans(float* a, int lda, float* b, int ldb, int m, int n) {
  constexpr int block_size = 6;
  // std::cout << " address of a: " << a << " lda: " << lda
  //           << " address of b: " << b << " ldb: " << ldb << " m: " << m
  //           << " n: " << n << std::endl;

  // N is the cols of matrix B
  int i = 0;
  for (; n - i >= block_size; i += block_size) {
    float* cur_a = a + i * lda;
    float* cur_b = b + i * ldb;

    // std::cout << "i " << i << std::endl;
    pack_no_trans_n6(cur_a, lda, cur_b, ldb, m, 6);
  }
  {
    // std::cout << "tail pack |i: " << i << " cols: " << n - i << std::endl;
    float* cur_a = a + i * lda;
    float* cur_b = b + i * ldb;

    pack_no_trans_nx(cur_a, lda, cur_b, ldb, m, 6, n - i);
  }

  // TODO (处理 tail)
}

void pack_no_trans_nx(float* a, const int lda, float* b, const int ldb,
                      const int m, const int n, int cols) {
  const int m8 = m / 8;
  const int m1 = m % 8;

  float* tmpa = a;
  float* tmpb = b;
  // cache

  float* aptrs[6];
  for (int i = 0; i < cols; i++) {
    aptrs[i] = tmpa + i * lda;
    // std::cout << "aptrs[i] " << aptrs[i][0] << " " << aptrs[i][511]
    //           << std::endl;
  }

  float adata[6][8];
  memset(adata, 0, sizeof(float) * 6 * 8);

  for (int m = 0; m < m8; m++) {
    // read
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < 8; j++) {
        adata[i][j] = aptrs[i][j];
      }
    }

    // write
    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < cols; ++i) {
        tmpb[j * 6 + i] = adata[i][j];
      }
    }

    // for (int i = 0; i < 48; i++) {
    //   std::cout << tmpb[i] << " ";
    // }
    // std::cout << std::endl;
    constexpr int vsize_in_bytes = 8;
    tmpb += 6 * vsize_in_bytes;

    // jump to another 8 float point values
    for (int i = 0; i < cols; i++) {
      aptrs[i] += vsize_in_bytes;
    }
  }
}

void pack_no_trans_n6(float* a, const int lda, float* b, const int ldb,
                      const int m, const int n) {
  const int m8 = m / 8;
  const int m1 = m % 8;
  const int block_size = n;

  float* tmpa = a;
  float* tmpb = b;
  float* a0 = tmpa + 0 * lda;
  float* a1 = tmpa + 1 * lda;
  float* a2 = tmpa + 2 * lda;
  float* a3 = tmpa + 3 * lda;
  float* a4 = tmpa + 4 * lda;
  float* a5 = tmpa + 5 * lda;

  for (int i = 0; i < m8; i++) {
    // std::cout << "pack_no_trans_n6| source " << std::endl;
    // std::vector<float*> vecs = {a0, a1, a2, a3, a4, a5};
    // for (auto vec : vecs)
    //   std::cout << vec[0] << " " << vec[1] << " " << vec[2] << " " << vec[3]
    //             << " " << vec[4] << " " << vec[5] << " " << vec[6] << " "
    //             << vec[7] << std::endl;
    __m256 v0 = _mm256_loadu_ps(a0);
    __m256 v1 = _mm256_loadu_ps(a1);
    __m256 v2 = _mm256_loadu_ps(a2);
    __m256 v3 = _mm256_loadu_ps(a3);
    __m256 v4 = _mm256_loadu_ps(a4);
    __m256 v5 = _mm256_loadu_ps(a5);

    __m256 unpack0 = _mm256_unpacklo_ps(v0, v1);
    __m256 unpack1 = _mm256_unpackhi_ps(v0, v1);
    __m256 unpack2 = _mm256_unpacklo_ps(v2, v3);
    __m256 unpack3 = _mm256_unpackhi_ps(v2, v3);
    __m256 unpack4 = _mm256_unpacklo_ps(v4, v5);
    __m256 unpack5 = _mm256_unpackhi_ps(v4, v5);

    __m256 shf0 = _mm256_shuffle_ps(unpack0, unpack2, 0x44);
    __m256 shf1 = _mm256_shuffle_ps(unpack4, unpack0, 0xe4);
    __m256 shf2 = _mm256_shuffle_ps(unpack2, unpack4, 0xee);
    __m256 shf3 = _mm256_shuffle_ps(unpack5, unpack1, 0xe4);
    __m256 shf4 = _mm256_shuffle_ps(unpack3, unpack5, 0xee);
    __m256 shf5 = _mm256_shuffle_ps(unpack1, unpack3, 0x44);

    __m128 low_shf1 = _mm256_castps256_ps128(shf1);
    __m256 res0 = _mm256_insertf128_ps(shf0, low_shf1, 0x1);
    __m256 res1 = _mm256_permute2f128_ps(shf0, shf1, 0x31);

    __m128 low_shf5 = _mm256_castps256_ps128(shf5);
    __m256 res2 = _mm256_insertf128_ps(shf2, low_shf5, 0x1);
    __m256 res3 = _mm256_permute2f128_ps(shf2, shf5, 0x31);

    __m128 low_shf4 = _mm256_castps256_ps128(shf4);
    __m256 res4 = _mm256_insertf128_ps(shf3, low_shf4, 0x1);
    __m256 res5 = _mm256_permute2f128_ps(shf3, shf4, 0x31);

    constexpr int vsize_in_bytes = 8;
    _mm256_storeu_ps(tmpb + 0 * vsize_in_bytes, res0);
    _mm256_storeu_ps(tmpb + 1 * vsize_in_bytes, res2);
    _mm256_storeu_ps(tmpb + 2 * vsize_in_bytes, res4);
    _mm256_storeu_ps(tmpb + 3 * vsize_in_bytes, res1);
    _mm256_storeu_ps(tmpb + 4 * vsize_in_bytes, res3);
    _mm256_storeu_ps(tmpb + 5 * vsize_in_bytes, res5);

    // std::cout << "pack_no_trans_n6| result " << std::endl;
    // std::vector<float*> vecb = {tmpb,      tmpb + 8,  tmpb + 16,
    //                             tmpb + 24, tmpb + 32, tmpb + 40};
    // for (auto vec : vecb)
    //   std::cout << vec[0] << " " << vec[1] << " " << vec[2] << " " << vec[3]
    //             << " " << vec[4] << " " << vec[5] << " " << vec[6] << " "
    //             << vec[7] << std::endl;

    tmpb += 6 * vsize_in_bytes;

    // jump to another 8 float point values
    a0 += vsize_in_bytes;
    a1 += vsize_in_bytes;
    a2 += vsize_in_bytes;
    a3 += vsize_in_bytes;
    a4 += vsize_in_bytes;
    a5 += vsize_in_bytes;
  }
}

void pack_trans_4x16(float* a, const int lda, float* b, const int ldb, int m,
                     int n);

//  pack block_size on leading dimension, t denotes transpose.
//  eg. input:   A MxN matrix in row major, so the storage-format is (M, N)
//      output:  B MxN matrix in col major(N-packed), so the storage-format is
//                    (divUp(N, block_size), M, block_size)
void pack_trans(float* a, int lda, float* b, int ldb, int m, int n) {
  // std::cout << " Address of a: " << a << " lda: " << lda
  //           << " Address of b: " << b << " ldb: " << ldb << " m: " << m
  //           << " n: " << n << std::endl;
  constexpr int block_size = 16;
  int i = 0;

  for (; i + 64 <= n; i += 64) {
    float* cur_a = a + i;
    float* cur_b = b + i * ldb;
    pack_trans_4x16(cur_a, lda, cur_b, ldb, m, block_size);
  }
}

void pack_trans_4x16(float* a, const int lda, float* b, const int ldb, int m,
                     int n) {
  const int m4 = m / 4;
  const int m1 = m % 1;
  const int block_size = 64;
  // const int ldb = ldb * 16 * sizeof(float);
  // lda = lda * 4
  const int ldbx16 = ldb * 16;  //(256 * 16)

  float* tmpa = a;
  float* ar0 = tmpa + 0 * lda;
  float* ar1 = tmpa + 1 * lda;
  float* ar2 = tmpa + 2 * lda;
  float* ar3 = tmpa + 3 * lda;

  float* tmpb = b;
  float* br0 = tmpb + 0 * ldbx16;
  float* br1 = tmpb + 1 * ldbx16;
  float* br2 = tmpb + 2 * ldbx16;
  float* br3 = tmpb + 3 * ldbx16;

  // 循环 256 / 4 = 64 次，每次 pack 4 x 16 = 64 个数据
  for (int i = 0; i < m4; ++i) {
    {
      __m256 v00 = _mm256_loadu_ps(ar0);
      __m256 v01 = _mm256_loadu_ps(ar0 + 8);
      __m256 v10 = _mm256_loadu_ps(ar1);
      __m256 v11 = _mm256_loadu_ps(ar1 + 8);
      __m256 v20 = _mm256_loadu_ps(ar2);
      __m256 v21 = _mm256_loadu_ps(ar2 + 8);
      __m256 v30 = _mm256_loadu_ps(ar3);
      __m256 v31 = _mm256_loadu_ps(ar3 + 8);

      _mm256_storeu_ps(br0 + 0, v00);
      _mm256_storeu_ps(br0 + 8, v01);
      _mm256_storeu_ps(br0 + 16, v10);
      _mm256_storeu_ps(br0 + 24, v11);
      _mm256_storeu_ps(br0 + 32, v20);
      _mm256_storeu_ps(br0 + 40, v21);
      _mm256_storeu_ps(br0 + 48, v30);
      _mm256_storeu_ps(br0 + 56, v31);
    }
    {
      __m256 v00 = _mm256_loadu_ps(ar0 + 16);
      __m256 v01 = _mm256_loadu_ps(ar0 + 24);
      __m256 v10 = _mm256_loadu_ps(ar1 + 16);
      __m256 v11 = _mm256_loadu_ps(ar1 + 24);
      __m256 v20 = _mm256_loadu_ps(ar2 + 16);
      __m256 v21 = _mm256_loadu_ps(ar2 + 24);
      __m256 v30 = _mm256_loadu_ps(ar3 + 16);
      __m256 v31 = _mm256_loadu_ps(ar3 + 24);

      _mm256_storeu_ps(br1 + 0, v00);
      _mm256_storeu_ps(br1 + 8, v01);
      _mm256_storeu_ps(br1 + 16, v10);
      _mm256_storeu_ps(br1 + 24, v11);
      _mm256_storeu_ps(br1 + 32, v20);
      _mm256_storeu_ps(br1 + 40, v21);
      _mm256_storeu_ps(br1 + 48, v30);
      _mm256_storeu_ps(br1 + 56, v31);
    }

    {
      __m256 v00 = _mm256_loadu_ps(ar0 + 32);
      __m256 v01 = _mm256_loadu_ps(ar0 + 40);
      __m256 v10 = _mm256_loadu_ps(ar1 + 32);
      __m256 v11 = _mm256_loadu_ps(ar1 + 40);
      __m256 v20 = _mm256_loadu_ps(ar2 + 32);
      __m256 v21 = _mm256_loadu_ps(ar2 + 40);
      __m256 v30 = _mm256_loadu_ps(ar3 + 32);
      __m256 v31 = _mm256_loadu_ps(ar3 + 40);

      _mm256_storeu_ps(br2 + 0, v00);
      _mm256_storeu_ps(br2 + 8, v01);
      _mm256_storeu_ps(br2 + 16, v10);
      _mm256_storeu_ps(br2 + 24, v11);
      _mm256_storeu_ps(br2 + 32, v20);
      _mm256_storeu_ps(br2 + 40, v21);
      _mm256_storeu_ps(br2 + 48, v30);
      _mm256_storeu_ps(br2 + 56, v31);
    }

    {
      __m256 v00 = _mm256_loadu_ps(ar0 + 48);
      __m256 v01 = _mm256_loadu_ps(ar0 + 56);
      __m256 v10 = _mm256_loadu_ps(ar1 + 48);
      __m256 v11 = _mm256_loadu_ps(ar1 + 56);
      __m256 v20 = _mm256_loadu_ps(ar2 + 48);
      __m256 v21 = _mm256_loadu_ps(ar2 + 56);
      __m256 v30 = _mm256_loadu_ps(ar3 + 48);
      __m256 v31 = _mm256_loadu_ps(ar3 + 56);

      _mm256_storeu_ps(br3 + 0, v00);
      _mm256_storeu_ps(br3 + 8, v01);
      _mm256_storeu_ps(br3 + 16, v10);
      _mm256_storeu_ps(br3 + 24, v11);
      _mm256_storeu_ps(br3 + 32, v20);
      _mm256_storeu_ps(br3 + 40, v21);
      _mm256_storeu_ps(br3 + 48, v30);
      _mm256_storeu_ps(br3 + 56, v31);
    }

    ar0 += 4 * lda;
    ar1 += 4 * lda;
    ar2 += 4 * lda;
    ar3 += 4 * lda;

    br0 += block_size;
    br1 += block_size;
    br2 += block_size;
    br3 += block_size;
  }

  // TODO 尾数处理
}

void col_major_micro_kernel_m16n6(const int K, const float alpha,
                                  const float* src_a, const int lda,
                                  const float* src_b, int ldb, const float beta,
                                  float* dst, int ldc);

void col_major_micro_kernel_m16n4(const int K, const float alpha,
                                  const float* src_a, const int lda,
                                  const float* src_b, int ldb, const float beta,
                                  float* dst_c, int ldc);

void col_major_micro_kernel_m16n2(const int K, const float alpha,
                                  const float* src_a, const int lda,
                                  const float* src_b, int ldb, const float beta,
                                  float* dst_c, int ldc);

void sgemm_block_n(int M, int N, int K, const float alpha, float* src_a,
                   int lda, float* src_b, int ldb, const float beta, float* dst,
                   int ldc) {
  int K_c = 256;
  int m_block = 16;

  for (int i = 0; i < M;) {
    int cur_m = std::min(int(M - i), 16);

    float* cur_a = src_a + DivDown(i, m_block) * K_c + i % m_block;
    float* cur_b = src_b;
    float* cur_c = dst + i;

    // std::cout << "sgemm_block_n: i = " << i << " cur_m: " << cur_m
    //           << " N: " << N << std::endl;
    switch (N) {
      case 6:
        col_major_micro_kernel_m16n6(K, alpha, cur_a, lda, cur_b, ldb, beta,
                                     cur_c, ldc);
        break;
      case 4:
        col_major_micro_kernel_m16n4(K, alpha, cur_a, lda, cur_b, ldb, beta,
                                     cur_c, ldc);
        break;
      case 2:
        col_major_micro_kernel_m16n2(K, alpha, cur_a, lda, cur_b, ldb, beta,
                                     cur_c, ldc);
        break;
      default:
        exit(0);
    }

    i += 16;
  }
}

// avx2 sgemm
void col_major_micro_kernel_m16n6(const int K, const float alpha,
                                  const float* src_a, const int lda,
                                  const float* src_b, int ldb, const float beta,
                                  float* dst_c, int ldc) {
  // load
  // micro kernel
  // (16, 8) * (8, 6) = (16, 6)
  const int mr = 8;
  const int nr = 6;

  constexpr int m_block_size = 16;
  constexpr int n_block_size = 6;

  // Load result matrix c (shape 16x6) into 12 x __m256 vector values
  __m256 c00 = _mm256_loadu_ps(dst_c + 0 * ldc);
  __m256 c01 = _mm256_loadu_ps(dst_c + 0 * ldc + 8);

  __m256 c10 = _mm256_loadu_ps(dst_c + 1 * ldc);
  __m256 c11 = _mm256_loadu_ps(dst_c + 1 * ldc + 8);

  __m256 c20 = _mm256_loadu_ps(dst_c + 2 * ldc);
  __m256 c21 = _mm256_loadu_ps(dst_c + 2 * ldc + 8);

  __m256 c30 = _mm256_loadu_ps(dst_c + 3 * ldc);
  __m256 c31 = _mm256_loadu_ps(dst_c + 3 * ldc + 8);

  __m256 c40 = _mm256_loadu_ps(dst_c + 4 * ldc);
  __m256 c41 = _mm256_loadu_ps(dst_c + 4 * ldc + 8);

  __m256 c50 = _mm256_loadu_ps(dst_c + 5 * ldc);
  __m256 c51 = _mm256_loadu_ps(dst_c + 5 * ldc + 8);

  // c = c * beta
  __m256 vbeta = _mm256_set1_ps(beta);

  c00 = _mm256_mul_ps(c00, vbeta);
  c01 = _mm256_mul_ps(c01, vbeta);

  c10 = _mm256_mul_ps(c10, vbeta);
  c11 = _mm256_mul_ps(c11, vbeta);

  c20 = _mm256_mul_ps(c20, vbeta);
  c21 = _mm256_mul_ps(c21, vbeta);

  c30 = _mm256_mul_ps(c30, vbeta);
  c31 = _mm256_mul_ps(c31, vbeta);

  c40 = _mm256_mul_ps(c40, vbeta);
  c41 = _mm256_mul_ps(c41, vbeta);

  c50 = _mm256_mul_ps(c50, vbeta);
  c51 = _mm256_mul_ps(c51, vbeta);

  // #pragma unroll
  for (int k = 0; k < K; ++k) {
    __m256 a0 = _mm256_loadu_ps(src_a);
    __m256 a1 = _mm256_loadu_ps(src_a + 8);

    __m256 vb = _mm256_broadcast_ss(src_b);
    c00 = _mm256_fmadd_ps(a0, vb, c00);
    c01 = _mm256_fmadd_ps(a1, vb, c01);

    vb = _mm256_broadcast_ss(src_b + 1);
    c10 = _mm256_fmadd_ps(a0, vb, c10);
    c11 = _mm256_fmadd_ps(a1, vb, c11);

    vb = _mm256_broadcast_ss(src_b + 2);
    c20 = _mm256_fmadd_ps(a0, vb, c20);
    c21 = _mm256_fmadd_ps(a1, vb, c21);

    vb = _mm256_broadcast_ss(src_b + 3);
    c30 = _mm256_fmadd_ps(a0, vb, c30);
    c31 = _mm256_fmadd_ps(a1, vb, c31);

    vb = _mm256_broadcast_ss(src_b + 4);
    c40 = _mm256_fmadd_ps(a0, vb, c40);
    c41 = _mm256_fmadd_ps(a1, vb, c41);

    vb = _mm256_broadcast_ss(src_b + 5);
    c50 = _mm256_fmadd_ps(a0, vb, c50);
    c51 = _mm256_fmadd_ps(a1, vb, c51);

    src_a += m_block_size;
    src_b += n_block_size;
  }

  __m256 valpha = _mm256_set1_ps(alpha);
  c00 = _mm256_mul_ps(c00, valpha);
  c01 = _mm256_mul_ps(c01, valpha);

  c10 = _mm256_mul_ps(c10, valpha);
  c11 = _mm256_mul_ps(c11, valpha);

  c20 = _mm256_mul_ps(c20, valpha);
  c21 = _mm256_mul_ps(c21, valpha);

  c30 = _mm256_mul_ps(c30, valpha);
  c31 = _mm256_mul_ps(c31, valpha);

  c40 = _mm256_mul_ps(c40, valpha);
  c41 = _mm256_mul_ps(c41, valpha);

  c50 = _mm256_mul_ps(c50, valpha);
  c51 = _mm256_mul_ps(c51, valpha);

  _mm256_storeu_ps(dst_c + 0 * ldc, c00);
  _mm256_storeu_ps(dst_c + 0 * ldc + 8, c01);

  _mm256_storeu_ps(dst_c + 1 * ldc, c10);
  _mm256_storeu_ps(dst_c + 1 * ldc + 8, c11);

  _mm256_storeu_ps(dst_c + 2 * ldc, c20);
  _mm256_storeu_ps(dst_c + 2 * ldc + 8, c21);

  _mm256_storeu_ps(dst_c + 3 * ldc, c30);
  _mm256_storeu_ps(dst_c + 3 * ldc + 8, c31);

  _mm256_storeu_ps(dst_c + 4 * ldc, c40);
  _mm256_storeu_ps(dst_c + 4 * ldc + 8, c41);

  _mm256_storeu_ps(dst_c + 5 * ldc, c50);
  _mm256_storeu_ps(dst_c + 5 * ldc + 8, c51);
}

void col_major_micro_kernel_m16n4(const int K, const float alpha,
                                  const float* src_a, const int lda,
                                  const float* src_b, int ldb, const float beta,
                                  float* dst_c, int ldc) {
  // load
  // micro kernel
  // (16, 8) * (8, 6) = (16, 6)
  const int mr = 8;
  const int nr = 6;

  constexpr int m_block_size = 16;
  constexpr int n_block_size = 6;

  // Load result matrix c (shape 16x6) into 12 x __m256 vector values
  __m256 c00 = _mm256_loadu_ps(dst_c + 0 * ldc);
  __m256 c01 = _mm256_loadu_ps(dst_c + 0 * ldc + 8);

  __m256 c10 = _mm256_loadu_ps(dst_c + 1 * ldc);
  __m256 c11 = _mm256_loadu_ps(dst_c + 1 * ldc + 8);

  __m256 c20 = _mm256_loadu_ps(dst_c + 2 * ldc);
  __m256 c21 = _mm256_loadu_ps(dst_c + 2 * ldc + 8);

  __m256 c30 = _mm256_loadu_ps(dst_c + 3 * ldc);
  __m256 c31 = _mm256_loadu_ps(dst_c + 3 * ldc + 8);

  // c = c * beta
  __m256 vbeta = _mm256_set1_ps(beta);

  c00 = _mm256_mul_ps(c00, vbeta);
  c01 = _mm256_mul_ps(c01, vbeta);

  c10 = _mm256_mul_ps(c10, vbeta);
  c11 = _mm256_mul_ps(c11, vbeta);

  c20 = _mm256_mul_ps(c20, vbeta);
  c21 = _mm256_mul_ps(c21, vbeta);

  c30 = _mm256_mul_ps(c30, vbeta);
  c31 = _mm256_mul_ps(c31, vbeta);

  // #pragma unroll
  for (int k = 0; k < K; ++k) {
    __m256 a0 = _mm256_loadu_ps(src_a);
    __m256 a1 = _mm256_loadu_ps(src_a + 8);

    // std::cout << "srca: " << src_a[0] << " " << src_a[15] << std::endl;

    // std::cout << "srcb: " << src_b[0] << " " << src_b[1] << " " << src_b[2]
    //           << " " << src_b[3] << " " << src_b[4] << " " << src_b[5]
    //           << std::endl;

    __m256 vb = _mm256_broadcast_ss(src_b);
    c00 = _mm256_fmadd_ps(a0, vb, c00);
    c01 = _mm256_fmadd_ps(a1, vb, c01);

    vb = _mm256_broadcast_ss(src_b + 1);
    c10 = _mm256_fmadd_ps(a0, vb, c10);
    c11 = _mm256_fmadd_ps(a1, vb, c11);

    vb = _mm256_broadcast_ss(src_b + 2);
    c20 = _mm256_fmadd_ps(a0, vb, c20);
    c21 = _mm256_fmadd_ps(a1, vb, c21);

    vb = _mm256_broadcast_ss(src_b + 3);
    c30 = _mm256_fmadd_ps(a0, vb, c30);
    c31 = _mm256_fmadd_ps(a1, vb, c31);

    src_a += m_block_size;
    src_b += n_block_size;
  }

  __m256 valpha = _mm256_set1_ps(alpha);
  c00 = _mm256_mul_ps(c00, valpha);
  c01 = _mm256_mul_ps(c01, valpha);

  c10 = _mm256_mul_ps(c10, valpha);
  c11 = _mm256_mul_ps(c11, valpha);

  c20 = _mm256_mul_ps(c20, valpha);
  c21 = _mm256_mul_ps(c21, valpha);

  c30 = _mm256_mul_ps(c30, valpha);
  c31 = _mm256_mul_ps(c31, valpha);

  _mm256_storeu_ps(dst_c + 0 * ldc, c00);
  _mm256_storeu_ps(dst_c + 0 * ldc + 8, c01);

  _mm256_storeu_ps(dst_c + 1 * ldc, c10);
  _mm256_storeu_ps(dst_c + 1 * ldc + 8, c11);

  _mm256_storeu_ps(dst_c + 2 * ldc, c20);
  _mm256_storeu_ps(dst_c + 2 * ldc + 8, c21);

  _mm256_storeu_ps(dst_c + 3 * ldc, c30);
  _mm256_storeu_ps(dst_c + 3 * ldc + 8, c31);
}

void col_major_micro_kernel_m16n2(const int K, const float alpha,
                                  const float* src_a, const int lda,
                                  const float* src_b, int ldb, const float beta,
                                  float* dst_c, int ldc) {
  // load
  // micro kernel
  // (16, 8) * (8, 6) = (16, 6)
  const int mr = 8;
  const int nr = 6;

  constexpr int m_block_size = 16;
  constexpr int n_block_size = 6;

  // Load result matrix c (shape 16x6) into 12 x __m256 vector values
  __m256 c00 = _mm256_loadu_ps(dst_c + 0 * ldc);
  __m256 c01 = _mm256_loadu_ps(dst_c + 0 * ldc + 8);

  __m256 c10 = _mm256_loadu_ps(dst_c + 1 * ldc);
  __m256 c11 = _mm256_loadu_ps(dst_c + 1 * ldc + 8);

  // c = c * beta
  __m256 vbeta = _mm256_set1_ps(beta);

  c00 = _mm256_mul_ps(c00, vbeta);
  c01 = _mm256_mul_ps(c01, vbeta);

  c10 = _mm256_mul_ps(c10, vbeta);
  c11 = _mm256_mul_ps(c11, vbeta);

  // #pragma unroll
  for (int k = 0; k < K; ++k) {
    __m256 a0 = _mm256_loadu_ps(src_a);
    __m256 a1 = _mm256_loadu_ps(src_a + 8);

    __m256 vb = _mm256_broadcast_ss(src_b);
    c00 = _mm256_fmadd_ps(a0, vb, c00);
    c01 = _mm256_fmadd_ps(a1, vb, c01);

    vb = _mm256_broadcast_ss(src_b + 1);
    c10 = _mm256_fmadd_ps(a0, vb, c10);
    c11 = _mm256_fmadd_ps(a1, vb, c11);

    src_a += m_block_size;
    src_b += n_block_size;
  }

  __m256 valpha = _mm256_set1_ps(alpha);
  c00 = _mm256_mul_ps(c00, valpha);
  c01 = _mm256_mul_ps(c01, valpha);

  c10 = _mm256_mul_ps(c10, valpha);
  c11 = _mm256_mul_ps(c11, valpha);

  _mm256_storeu_ps(dst_c + 0 * ldc, c00);
  _mm256_storeu_ps(dst_c + 0 * ldc + 8, c01);

  _mm256_storeu_ps(dst_c + 1 * ldc, c10);
  _mm256_storeu_ps(dst_c + 1 * ldc + 8, c11);
}

// Col major sgemm avx2
void avx2_col_major_sgemm(int M, int N, int K, float alpha, float* A, int lda,
                          float* B, int ldb, float beta, float* C, int ldc) {
  // Assert
  if (alpha == 0) return;

  float beta_div_alpha = beta / alpha;

  constexpr int Mc = 64;
  constexpr int Kc = 256;

  constexpr int mr = 16;
  constexpr int nr = 6;

  // Cache a is 64 x 256
  float* pack_a = (float*)_mm_malloc(Mc * Kc * sizeof(float), 32);
  // Cache b is 256 x N
  float* pack_b = (float*)_mm_malloc(Kc * DivUp(N, nr) * sizeof(float), 32);

  // std::cout << "Floats of cache pack_a " << Mc * Kc << std::endl;
  // std::cout << "Floats of cache pack_b " << Kc * DivUp(N, nr) << std::endl;

  float* tmp_pack_a = pack_a;
  float* tmp_pack_b = pack_b;
  for (int k = 0; k < K; k += Kc) {
    float cur_beta = 1.0 / alpha;
    if (k == 0) cur_beta = beta_div_alpha;

    int cur_k = std::min(K - k, Kc);

    // std::cout << "Pack matrix of B: cur_k = " << cur_k << std::endl;
    // jump to k-th row of matrix B
    pack_no_trans(B + k, ldb, tmp_pack_b, Kc, cur_k, N);

    for (int i = 0; i < M; i += Mc) {
      int cur_m = std::min(M - i, Mc);
      //      std::cout << "Pack matrix of A: cur_m = " << cur_m << std::endl;
      pack_trans(A + i + k * lda, lda, tmp_pack_a, Kc, cur_k, cur_m);

      for (int j = 0; j < N;) {
        int cur_n = std::min(int(N - j), nr);
        float* cur_c = C + i + j * ldc;

        float* packed_cur_b = tmp_pack_b + DivDown(j, nr) * Kc + j % nr;

        sgemm_block_n(cur_m, cur_n, cur_k, alpha, tmp_pack_a, lda, packed_cur_b,
                      ldb, cur_beta, cur_c, ldc);
        j += cur_n;
      }
    }
  }

  _mm_free(pack_a);
  _mm_free(pack_b);
}

// void sgemm(char transa, char transb, int M, int N, int K, float alpha,
//            const float* A, int lda, const float* B, int ldb, float beta,
//            float* C, int ldc) {
//   avx2_col_major_sgemm('N', 'N', M, N, K, alpha, A, lda, B, ldb, beta, C,
//   ldc);
// }

}  // namespace x86
}  // namespace device
}  // namespace tars