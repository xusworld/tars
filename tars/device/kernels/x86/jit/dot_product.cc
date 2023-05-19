#include "dot_product.h"

// Copy From weihuali
namespace kernels {
namespace cpu {
namespace exp {

// DotProduct(float* x, float* y, int n, float* output, int m);
//              rdi,         rsi,   rdx,       rcx,   r8
void DotProductImpl::GenerateCode() {
  Prolog();
  uni_vzeroupper();

  constexpr int64_t kStep = sizeof(float) * kUnroll * kPack;

  for (int k = 0; k < kUnroll; ++k) {
    vpxor(Xbyak::Ymm(k), Xbyak::Ymm(k), Xbyak::Ymm(k));
  }

  Xbyak::Label unroll_loop, unroll_end, left_loop, left_end;

  test(reg_m_, reg_m_);
  jng(unroll_end, T_NEAR);

  mov(reg_loop_, reg_m_);

  L(unroll_loop);
  {
    // unrool loop
    for (int k = 0; k < kUnroll; ++k) {
      vmovaps(px_, ptr[reg_x_ + k * sizeof(float) * kPack]);
      vfmadd231ps(Xbyak::Ymm(k), px_, ptr[reg_y_ + k * sizeof(float) * kPack]);
    }

    add(reg_x_, kStep);
    add(reg_y_, kStep);
    sub(reg_loop_, 1);
    jnz(unroll_loop);
  }
  L(unroll_end);

  // left loop
  // Best use padded data for performance!
  imul(reg_m_, reg_m_, kUnroll);
  sub(reg_n_, reg_m_);

  // when no left
  test(reg_n_, reg_n_);
  jng(left_end, T_NEAR);

  L(left_loop);
  {
    vmovaps(px_, ptr[reg_x_]);
    vmovaps(py_, ptr[reg_y_]);
    // How to use different ymmm register?
    vfmadd231ps(Xbyak::Ymm(0), px_, py_);
    add(reg_x_, sizeof(float) * kPack);
    add(reg_y_, sizeof(float) * kPack);

    sub(reg_n_, 1);
    jnz(left_loop);
  }
  L(left_end);

  int m = kUnroll;
  while (m > 1) {
    m /= 2;
    for (int k = 0; k < m; ++k) {
      vaddps(Xbyak::Ymm(k), Xbyak::Ymm(k), Xbyak::Ymm(k + m));
    }
  }

  vextractf128(xmm1, ymm0, 1);
  vaddps(xmm0, xmm1);
  vhaddps(xmm0, xmm0);
  vhaddps(xmm0, xmm0);
  vmovss(ptr[rcx], xmm0);

  Epilog();
}  // namespace exp

float DotProductImpl::operator()(float *x, float *y, int n) {
  assert(jit_kernel_ && "Forget to set param n");

  float result;

  n /= kPack;

  // unrolled loop number
  int m = n / kUnroll;

  // call jit kernel
  JITCodeGenerator::operator()(x, y, n, &result, m);

  return result;
}

}  // namespace exp
}  // namespace cpu
}  // namespace kernels
