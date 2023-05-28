#include <stdlib.h>

#include <cstring>

#include "sgemm.h"

namespace tars {
namespace device {
namespace x86 {

// 原始算法
void NaiveGemm(const float* A, const float* B, float* C, const int M,
               const int N, const int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// 优化访存
void NaiveGemmOptim1(const float* A, const float* B, float* C, const int M,
                     const int N, const int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

void NaiveGemmColMajor(const float* A, const float* B, float* C, const int M,
                       const int N, const int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i + j * M] += A[i + k * M] * B[j * K + k];
      }
    }
  }
}

void naive_col_major_sgemm(const float* A, const float* B, float* C,
                           const int M, const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; k++) {
        C[m + n * M] += A[m + k * M] * B[n * K + k];
      }
    }
  }
}

void naive_col_major_optimize_sgemm(const float* A, const float* B, float* C,
                                    const int M, const int N, const int K) {
  for (int k = 0; k < K; k++) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        C[m + n * M] += A[m + k * M] * B[n * K + k];
      }
    }
  }
}

/* Strassen Algorithm
 * A a M * K matrix
 * B a K * N matrix
 * C a M * N matrix
 * C = A * B
 * M1 = (A11 + A22) * (B11 + B22)
 * M2 = (A21 + A22) * B11
 * M3 = A11 * (B12 - B22)
 * M4 = A22 * (B21 - B11)
 * M5 = (A11 + A12) * B22
 * M6 = (A21 - A11) * (B11 + B12)
 * M7 = (A12 - A22) * (B21 + B22)
 * C11 = M1 + M4 - M5 + M7
 * C12 = M3 + M5
 * C21 = M2 + M4
 * C22 = M1 - M2 + M3 + M6
 */
void StrassenGemm(float* A, float* B, float* C, const int M, const int N,
                  const int K) {
  if ((M <= 2) || M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
    return NaiveGemm(A, B, C, M, N, K);
  }

  int offset = 0;
  // M1 = (A11 + A22) * (B11 + B22)
  float* M1 = (float*)malloc((M / 2) * (N / 2) * sizeof(float));
  {
    // M1_0 = (A11 + A22)
    float* M1_0 = (float*)malloc((M / 2) * (K / 2) * sizeof(float));
    offset = M * K / 2 + K / 2;
    for (int i = 0; i < M / 2; i++) {
      for (int j = 0; j < K / 2; j++) {
        const int baseIdx = i * K + j;
        M1_0[i * K / 2 + j] = A[baseIdx] + A[baseIdx + offset];
      }
    }
    // M1_1 = (B11 + B22)
    float* M1_1 = (float*)malloc((K / 2) * (N / 2) * sizeof(float));
    offset = K * N / 2 + N / 2;
    for (int i = 0; i < K / 2; i++) {
      for (int j = 0; j < N / 2; j++) {
        const int baseIdx = i * N + j;
        M1_1[i * N / 2 + j] = B[baseIdx] + B[baseIdx + offset];
      }
    }
    StrassenGemm(&M1_0[0], &M1_1[0], &M1[0], M / 2, N / 2, K / 2);

    free(M1_0);
    M1_0 = NULL;
    free(M1_1);
    M1_1 = NULL;
  }

  // M2 = (A21+A22)*B11
  float* M2 = (float*)malloc((M / 2) * (N / 2) * sizeof(float));
  {
    // M2_0 = (A21+A22)
    float* M2_0 = (float*)malloc((M / 2) * (K / 2) * sizeof(float));
    offset = K / 2;
    for (int i = M / 2; i < M; i++) {
      for (int j = 0; j < K / 2; j++) {
        const int baseIdx = i * K + j;
        M2_0[(i - M / 2) * K / 2 + j] = A[baseIdx] + A[baseIdx + offset];
      }
    }
    // M2_1 = B11
    float* M2_1 = (float*)malloc((K / 2) * (N / 2) * sizeof(float));
    for (int i = 0; i < K / 2; i++) {
      for (int j = 0; j < N / 2; j++) {
        M2_1[i * N / 2 + j] = B[i * N + j];
      }
    }
    StrassenGemm(&M2_0[0], &M2_1[0], &M2[0], M / 2, N / 2, K / 2);

    free(M2_0);
    M2_0 = NULL;
    free(M2_1);
    M2_1 = NULL;
  }

  // M3 = A11*(B12-B22)
  float* M3 = (float*)malloc((M / 2) * (N / 2) * sizeof(float));
  {
    // M3_0 = A11
    float* M3_0 = (float*)malloc((M / 2) * (K / 2) * sizeof(float));
    for (int i = 0; i < M / 2; i++) {
      for (int j = 0; j < K / 2; j++) {
        M3_0[i * K / 2 + j] = A[i * K + j];
      }
    }
    // M3_1 = (B12-B22)
    float* M3_1 = (float*)malloc((K / 2) * (N / 2) * sizeof(float));
    offset = K * N / 2;
    for (int i = 0; i < K / 2; i++) {
      for (int j = N / 2; j < N; j++) {
        const int baseIdx = i * N + j;
        M3_1[i * N / 2 + j - N / 2] = B[baseIdx] - B[baseIdx + offset];
      }
    }
    StrassenGemm(&M3_0[0], &M3_1[0], &M3[0], M / 2, N / 2, K / 2);

    free(M3_0);
    M3_0 = NULL;
    free(M3_1);
    M3_1 = NULL;
  }

  // M4 = A22*(B21-B11)
  float* M4 = (float*)malloc((M / 2) * (N / 2) * sizeof(float));
  {
    // M4_0 = A22
    float* M4_0 = (float*)malloc((M / 2) * (K / 2) * sizeof(float));
    for (int i = M / 2; i < M; i++) {
      for (int j = K / 2; j < K; j++) {
        M4_0[(i - M / 2) * K / 2 + j - K / 2] = A[i * K + j];
      }
    }
    // M4_1 = (B21-B11)
    float* M4_1 = (float*)malloc((K / 2) * (N / 2) * sizeof(float));
    offset = N * K / 2;
    for (int i = 0; i < K / 2; i++) {
      for (int j = 0; j < N / 2; j++) {
        const int baseIdx = i * N + j;
        M4_1[i * N / 2 + j] = B[baseIdx + offset] - B[baseIdx];
      }
    }
    StrassenGemm(&M4_0[0], &M4_1[0], &M4[0], M / 2, N / 2, K / 2);

    free(M4_0);
    M4_0 = NULL;
    free(M4_1);
    M4_1 = NULL;
  }

  // M5 = (A11+A12)*B22
  float* M5 = (float*)malloc((M / 2) * (N / 2) * sizeof(float));
  {
    // M5_0 = (A11+A12)
    float* M5_0 = (float*)malloc((M / 2) * (K / 2) * sizeof(float));
    offset = K / 2;
    for (int i = 0; i < M / 2; i++) {
      for (int j = 0; j < K / 2; j++) {
        const int baseIdx = i * K + j;
        M5_0[i * K / 2 + j] = A[baseIdx] + A[baseIdx + offset];
      }
    }
    // M5_1 = B22
    float* M5_1 = (float*)malloc((K / 2) * (N / 2) * sizeof(float));
    offset = N * K / 2 + N / 2;
    for (int i = 0; i < K / 2; i++) {
      for (int j = 0; j < N / 2; j++) {
        M5_1[i * N / 2 + j] = B[i * N + j + offset];
      }
    }
    StrassenGemm(&M5_0[0], &M5_1[0], &M5[0], M / 2, N / 2, K / 2);

    free(M5_0);
    M5_0 = NULL;
    free(M5_1);
    M5_1 = NULL;
  }

  // M6 = (A21-A11)*(B11+B12)
  float* M6 = (float*)malloc((M / 2) * (N / 2) * sizeof(float));
  {
    // M6_0 = (A21-A11)
    float* M6_0 = (float*)malloc((M / 2) * (K / 2) * sizeof(float));
    offset = K * M / 2;
    for (int i = 0; i < M / 2; i++) {
      for (int j = 0; j < K / 2; j++) {
        const int baseIdx = i * K + j;
        M6_0[i * K / 2 + j] = A[baseIdx + offset] - A[baseIdx];
      }
    }
    // M6_1 = (B11+B12)
    float* M6_1 = (float*)malloc((K / 2) * (N / 2) * sizeof(float));
    offset = N / 2;
    for (int i = 0; i < K / 2; i++) {
      for (int j = 0; j < N / 2; j++) {
        const int baseIdx = i * N + j;
        M6_1[i * N / 2 + j] = B[baseIdx] + B[baseIdx + offset];
      }
    }
    StrassenGemm(&M6_0[0], &M6_1[0], &M6[0], M / 2, N / 2, K / 2);

    free(M6_0);
    M6_0 = NULL;
    free(M6_1);
    M6_1 = NULL;
  }

  // M7 = (A12-A22)*(B21+B22)
  float* M7 = (float*)malloc((M / 2) * (N / 2) * sizeof(float));
  {
    // M7_0 = (A12-A22)
    float* M7_0 = (float*)malloc((M / 2) * (K / 2) * sizeof(float));
    offset = M * K / 2;
    for (int i = 0; i < M / 2; i++) {
      for (int j = K / 2; j < K; j++) {
        const int baseIdx = i * K + j;
        M7_0[i * K / 2 + j - K / 2] = A[baseIdx] - A[baseIdx + offset];
      }
    }
    // M7_1 = (B21+B22)
    float* M7_1 = (float*)malloc((K / 2) * (N / 2) * sizeof(float));
    offset = N / 2;
    for (int i = K / 2; i < K; i++) {
      for (int j = 0; j < N / 2; j++) {
        const int baseIdx = i * N + j;
        M7_1[(i - K / 2) * N / 2 + j] = B[baseIdx] + B[baseIdx + offset];
      }
    }
    StrassenGemm(&M7_0[0], &M7_1[0], &M7[0], M / 2, N / 2, K / 2);

    free(M7_0);
    M7_0 = NULL;
    free(M7_1);
    M7_1 = NULL;
  }

  for (int i = 0; i < M / 2; i++) {
    for (int j = 0; j < N / 2; j++) {
      const int idx = i * N / 2 + j;
      // C11 = M1+M4-M5+M7
      C[i * N + j] = M1[idx] + M4[idx] - M5[idx] + M7[idx];
      // C12 = M3+M5
      C[i * N + j + N / 2] = M3[idx] + M5[idx];
      // C21 = M2+M4
      C[(i + M / 2) * N + j] = M2[idx] + M4[idx];
      // C22 = M1-M2+M3+M6
      C[(i + M / 2) * N + j + N / 2] = M1[idx] - M2[idx] + M3[idx] + M6[idx];
    }
  }
  free(M1);
  M1 = NULL;
  free(M2);
  M2 = NULL;
  free(M3);
  M3 = NULL;
  free(M4);
  M4 = NULL;
  free(M5);
  M5 = NULL;
  free(M6);
  M6 = NULL;
  free(M7);
  M7 = NULL;
}

}  // namespace x86
}  // namespace device
}  // namespace tars