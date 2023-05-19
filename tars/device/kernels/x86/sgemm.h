#pragma once

namespace ace {
namespace device {
namespace x86 {

/*

C := alpha * A  * B + beta * C

- A is a M x K matrix
- B is a K x N matrix
- C is a M x N matrix

The matrices are assumed to be stored in row-major order (the elements in each
of the matrix rows are contiguous in memory).

- transa	Transposition flag for matrix A: 'N' or 'n' means A is not
transposed, and 'T' or 't' means that A is transposed.
- transb	Transposition flag for matrix B: 'N' or 'n' means B is not
transposed, and 'T' or 't' means that B is transposed.
- M	The M dimension.
- N	The N dimension.
- K	The K dimension.
- alpha	The alpha parameter that is used to scale the product of matrices A and
B.
- A	A pointer to the A matrix data.
- lda	The leading dimension for the matrix A.
- B	A pointer to the B matrix data.
- ldb	The leading dimension for the matrix B.
- beta	The beta parameter that is used to scale the matrix C.
- C	A pointer to the C matrix data.
- ldc	The leading dimension for the matrix C.
*/
void sgemm(char transa, char transb, int M, int N, int K, float alpha,
           const float* A, int lda, const float* B, int ldb, float beta,
           float* C, int ldc);

// Naive GEMM algorithm
// Matrix A with a shape (M, K), Matrix B with a shape (K, N) and
// Matrix A with a shape (M, N)
void NaiveGemm(const float* A, const float* B, float* C, const int M,
               const int N, const int K);

void NaiveGemmOptim1(const float* A, const float* B, float* C, const int M,
                     const int N, const int K);

void NaiveGemmColMajor(const float* A, const float* B, float* C, const int M,
                       const int N, const int K);

void StrassenGemm(float* A, float* B, float* C, const int M, const int N,
                  const int K);

}  // namespace x86
}  // namespace device
}  // namespace ace